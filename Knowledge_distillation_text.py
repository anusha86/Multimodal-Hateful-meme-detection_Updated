"""
Fine-tune DistilBERT on ~743 meme captions
(offensive vs non-offensive), ROCm-ready.

Folders
───────
/home/agam/Downloads/g/distilbert/log     – logs + metrics CSV
/home/agam/Downloads/g/distilbert/graphs  – curves + confusion matrices
/home/agam/Downloads/g/distilbert/pt      – best model (.pt)
"""

import os
import random
import logging
import traceback

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix
)

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import Adam
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ────────────────────────── Paths ──────────────────────────
BASE_DIR  = "/home/agam/Downloads/g/distilbert"
LOG_DIR   = os.path.join(BASE_DIR, "log")
GRAPH_DIR = os.path.join(BASE_DIR, "graphs")
PT_DIR    = os.path.join(BASE_DIR, "pt")
for d in (LOG_DIR, GRAPH_DIR, PT_DIR):
    os.makedirs(d, exist_ok=True)

TRAIN_CSV = "/home/agam/Downloads/g/Split Dataset/Testing_meme_dataset.csv"
VAL_CSV   = "/home/agam/Downloads/g/Split Dataset/Validation_meme_dataset.csv"
MODEL_OUT = os.path.join(PT_DIR, "distilbert_best.pt")

# ───────────────────────── Logging ─────────────────────────
LOG_FILE = os.path.join(LOG_DIR, "distilbert_training.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ],
)
logger = logging.getLogger()

# ───────────────────── Reproducibility ─────────────────────
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ───────────────────────── Device ─────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(
    "Running on %s (%s build)",
    torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
    "ROCm" if device.type == "cuda" else "native"
)

scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
AMP = lambda: torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda"))

# ─────────────────────── Utilities ────────────────────────
def map_label(x: str) -> int:
    x = str(x).strip().lower()
    return 1 if ("offensive" in x and "non" not in x) else 0

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
MAX_LEN = 128

class MemeTextDS(Dataset):
    def __init__(self, df):
        self.texts = df["sentence"].tolist()
        self.labels = df["label"].astype(int).tolist()
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        enc = tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=MAX_LEN,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def metrics_dict(y_true, y_pred):
    return dict(
        acc  = accuracy_score(y_true, y_pred),
        prec = precision_score(y_true, y_pred, average="weighted", zero_division=0),
        rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0),
        f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0),
        mcc  = matthews_corrcoef(y_true, y_pred),
        cm   = confusion_matrix(y_true, y_pred),
    )

def log_metrics(phase, loss, m):
    logger.info(
        f"{phase} - Loss {loss:.4f} | "
        f"Acc {m['acc']:.4f} Prec {m['prec']:.4f} "
        f"Rec {m['rec']:.4f} F1 {m['f1']:.4f} MCC {m['mcc']:.4f}"
    )

def save_curve(hist, key, ylab):
    xs = range(1, len(hist["epoch"]) + 1)
    plt.figure(figsize=(8,6))
    plt.plot(xs, hist[f"train_{key}"], "o-", label="Train")
    plt.plot(xs, hist[f"val_{key}"],   "o-", label="Val")
    plt.xlabel("Epoch"); plt.ylabel(ylab); plt.title(f"{ylab} Curve"); plt.legend()
    fp = os.path.join(GRAPH_DIR, f"distilbert_{key}_curve.png")
    plt.savefig(fp); plt.close(); logger.info("Saved %s", fp)

def save_cm(cm, title, fname):
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-Off","Off"], yticklabels=["Non-Off","Off"])
    plt.title(title); plt.ylabel("True"); plt.xlabel("Pred")
    plt.tight_layout(); plt.savefig(fname); plt.close(); logger.info("Saved %s", fname)

# ──────────────────────── Training ────────────────────────
try:
    # load & preprocess data
    tr_df = pd.read_csv(TRAIN_CSV).dropna(subset=["sentence","label"])
    vl_df = pd.read_csv(VAL_CSV).dropna(subset=["sentence","label"])
    tr_df["label"] = tr_df["label"].apply(map_label)
    vl_df["label"] = vl_df["label"].apply(map_label)
    logger.info("Train head:\n%s", tr_df.head())
    logger.info("Val head:\n%s",   vl_df.head())

    # class balancing sampler
    cls_cnt = tr_df["label"].value_counts().to_dict()
    weights = tr_df["label"].apply(lambda c: 1/cls_cnt[c]).values
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    BATCH = 16
    train_loader = DataLoader(
        MemeTextDS(tr_df),
        batch_size=BATCH,
        sampler=sampler,
        num_workers=4,
        pin_memory=(device.type=="cuda"),
    )
    val_loader = DataLoader(
        MemeTextDS(vl_df),
        batch_size=BATCH,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type=="cuda"),
    )

    # model, optimizer, schedule, loss
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    ).to(device)

    
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(WARM_FRAC * len(train_loader) * EPOCHS),
        num_training_steps=len(train_loader) * EPOCHS,
    )
    loss_fn = nn.CrossEntropyLoss()

    history = {k: [] for k in [
        "epoch","train_loss","val_loss",
        "train_acc","val_acc","train_prec","val_prec",
        "train_rec","val_rec","train_f1","val_f1",
        "train_mcc","val_mcc"
    ]}

    best_acc, best_cm, no_imp = 0.0, None, 0

    @torch.no_grad()
    def evaluate(loader):
        model.eval()
        total_loss, preds, trues = 0.0, [], []
        with AMP():
            for batch in loader:
                b = {k: v.to(device) for k, v in batch.items()}
                out = model(
                    input_ids=b["input_ids"],
                    attention_mask=b["attention_mask"]
                )
                logits = out.logits
                total_loss += loss_fn(logits, b["labels"]).item() * logits.size(0)
                preds.extend(logits.argmax(1).cpu().numpy())
                trues.extend(b["labels"].cpu().numpy())
        return total_loss / len(loader.dataset), metrics_dict(trues, preds)

    # epoch loop
    for epoch in range(1, EPOCHS+1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            b = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            with AMP():
                out  = model(
                    input_ids=b["input_ids"],
                    attention_mask=b["attention_mask"]
                )
                loss = loss_fn(out.logits, b["labels"])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += loss.item() * b["labels"].size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_loss, val_met   = evaluate(val_loader)
        train_loss_eval, train_met = evaluate(train_loader)

        # record history
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss_eval)
        history["val_loss"].append(val_loss)
        for key in ("acc","prec","rec","f1","mcc"):
            history[f"train_{key}"].append(train_met[key])
            history[f"val_{key}"].append(val_met[key])

        logger.info("Epoch %d/%d", epoch, EPOCHS)
        log_metrics("Train", train_loss_eval, train_met)
        log_metrics("Val",   val_loss,       val_met)
        logger.info("Val CM:\n%s\n", val_met["cm"])

        # checkpoint
        if val_met["acc"] > best_acc:
            best_acc, no_imp = val_met["acc"], 0
            best_cm = val_met["cm"]
            torch.save(model.state_dict(), MODEL_OUT)
            save_cm(best_cm, "Best Confusion Matrix",
                    os.path.join(GRAPH_DIR, "distilbert_confmat_best.png"))
            logger.info("↑ New best %.4f saved to %s", best_acc, MODEL_OUT)
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                logger.info("Early stopping (no improve %d)", no_imp)
                break

    # post-run artifacts
    save_cm(val_met["cm"], "Final Confusion Matrix",
            os.path.join(GRAPH_DIR, "distilbert_confmat_final.png"))
    for key, lab in [
        ("loss","Loss"), ("acc","Accuracy"), ("prec","Precision"),
        ("rec","Recall"), ("f1","F1"), ("mcc","MCC")
    ]:
        save_curve(history, key, lab)

    # save per-epoch metrics
    hist_df = pd.DataFrame(history).round(4)
    hist_df.to_csv(os.path.join(LOG_DIR, "metrics_history.csv"), index=False)
    logger.info("Saved per-epoch metrics to %s", os.path.join(LOG_DIR, "metrics_history.csv"))

    logger.info("Training finished. Best Val Acc = %.4f", best_acc)
    print(f"Best validation accuracy: {best_acc:.4f}")

except Exception:
    logger.error("Exception:\n%s", traceback.format_exc())
    raise
