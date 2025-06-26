
import os, random, logging, traceback
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torchvision import transforms
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             confusion_matrix, precision_score, recall_score)
import timm
from timm.loss import SoftTargetCrossEntropy
from timm.data import Mixup

BASE_DIR   = "/home/agam/Downloads/g/swint"
LOG_DIR    = os.path.join(BASE_DIR, "log")
GRAPH_DIR  = os.path.join(BASE_DIR, "graphs")
PT_DIR     = os.path.join(BASE_DIR, "pt")
for d in (LOG_DIR, GRAPH_DIR, PT_DIR):
    os.makedirs(d, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "swin_training.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
logger = logging.getLogger()

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Running on %s (%s build)",
            torch.cuda.get_device_name(0) if device.type=="cuda" else "CPU",
            "ROCm" if device.type=="cuda" else "native")
scaler = torch.amp.GradScaler(enabled=device.type=="cuda")
AMP = lambda : torch.amp.autocast(device_type='cuda', enabled=device.type=="cuda")

TRAIN_CSV = "/home/agam/Downloads/g/Split Dataset/Testing_meme_dataset.csv"
VAL_CSV   = "/home/agam/Downloads/g/Split Dataset/Validation_meme_dataset.csv"
IMAGE_DIR = "/home/agam/Downloads/g/Labelled Images"
MODEL_OUT = os.path.join(PT_DIR, "swin_tiny_best.pt")

def map_label(x: str) -> int:
    x = str(x).strip().lower()
    return 1 if ("offensive" in x and "non" not in x) else 0

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.25, 0.25, 0.25, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])
eval_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

class MemeDataset(Dataset):
    def __init__(self, df, img_dir, tf):
        self.df, self.dir, self.tf = df.reset_index(drop=True), img_dir, tf
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.dir, row["image_name"])).convert("RGB")
        return self.tf(img), int(row["label"])

def metrics(y_true, y_pred):
    return dict(
        acc  = accuracy_score(y_true, y_pred),
        prec = precision_score(y_true, y_pred, average="weighted", zero_division=0),
        rec  = recall_score(y_true,  y_pred, average="weighted", zero_division=0),
        f1   = f1_score(y_true,     y_pred, average="weighted", zero_division=0),
        mcc  = matthews_corrcoef(y_true,   y_pred),
        cm   = confusion_matrix(y_true,    y_pred),
    )

def log_metrics(phase, loss, m):
    logger.info(
        f"{phase} - Loss {loss:.4f} | "
        f"Acc {m['acc']:.4f} Prec {m['prec']:.4f} "
        f"Rec {m['rec']:.4f} F1 {m['f1']:.4f} MCC {m['mcc']:.4f}"
    )

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

def save_curve(hist, key, ylab):
    xs = range(1, len(hist["epoch"])+1)
    plt.figure(figsize=(8,6))
    plt.plot(xs, hist[f"train_{key}"], "o-", label="Train")
    plt.plot(xs, hist[f"val_{key}"],   "o-", label="Val")
    plt.xlabel("Epoch"); plt.ylabel(ylab); plt.title(f"{ylab} Curve"); plt.legend()
    fp = os.path.join(GRAPH_DIR, f"swin_{key}_curve.png")
    plt.savefig(fp); plt.close(); logger.info("Saved %s", fp)

def save_cm(cm, title, fname):
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-Off","Off"], yticklabels=["Non-Off","Off"])
    plt.title(title); plt.ylabel("True"); plt.xlabel("Pred")
    plt.tight_layout(); plt.savefig(fname); plt.close(); logger.info("Saved %s", fname)

try:

    tr_df = pd.read_csv(TRAIN_CSV).dropna(subset=["label"])
    vl_df = pd.read_csv(VAL_CSV).dropna(subset=["label"])
    tr_df["label"] = tr_df["label"].apply(map_label)
    vl_df["label"] = vl_df["label"].apply(map_label)
    logger.info("Train head:\n%s", tr_df.head())
    logger.info("Val head:\n%s",   vl_df.head())

    cls_cnt = tr_df["label"].value_counts().to_dict()
    weights = tr_df["label"].apply(lambda c: 1/cls_cnt[c]).values
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    BATCH = 16
    train_ds = MemeDataset(tr_df, IMAGE_DIR, train_tf)
    val_ds   = MemeDataset(vl_df, IMAGE_DIR, eval_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH, sampler=sampler,
                              num_workers=4, pin_memory=(device.type=="cuda"),
                              drop_last=True)      
    train_eval   = DataLoader(train_ds, batch_size=BATCH, shuffle=False,
                              num_workers=4, pin_memory=(device.type=="cuda"))
    val_loader   = DataLoader(val_ds,  batch_size=BATCH, shuffle=False,
                              num_workers=4, pin_memory=(device.type=="cuda"))

    mixup_fn = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0,
                     label_smoothing=0.1, num_classes=2)

    model = timm.create_model("swin_tiny_patch4_window7_224",
                              pretrained=True, num_classes=2).to(device)

    for p in model.parameters(): p.requires_grad = False
    for p in model.head.parameters(): p.requires_grad = True

    PHASE1, PHASE2 = 10, 40
    EPOCHS, PATIENCE = PHASE1+PHASE2, 8

    opt   = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                  lr=3e-4, weight_decay=0.05)
    sched = get_cosine_schedule_with_warmup(
        opt, int(0.1*len(train_loader)*EPOCHS),
        len(train_loader)*EPOCHS)

    loss_train = SoftTargetCrossEntropy()
    loss_eval  = nn.CrossEntropyLoss()

    hist = {k: [] for k in
            ["epoch","train_loss","val_loss",
             "train_acc","val_acc","train_prec","val_prec",
             "train_rec","val_rec","train_f1","val_f1",
             "train_mcc","val_mcc"]}

    best_acc, best_cm, no_imp = 0.0, None, 0

    @torch.no_grad()
    def evaluate(loader):
        model.eval()
        tl, yp, yt = 0.0, [], []
        with AMP():
            for x,y in loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                out  = model(x)
                tl  += loss_eval(out, y).item() * x.size(0)
                yp.extend(out.argmax(1).cpu().numpy()); yt.extend(y.cpu().numpy())
        return tl/len(loader.dataset), metrics(yt, yp)

    for epoch in range(1, EPOCHS+1):
        model.train(); running = 0.0
        for x,y in train_loader:

            if x.size(0) & 1:          
                x, y = x[:-1], y[:-1]
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            x, y = mixup_fn(x, y)
            opt.zero_grad()
            with AMP():
                out = model(x); loss = loss_train(out, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            running += loss.item()*x.size(0)

        tr_loss = running/len(train_loader.dataset)
        tl, tmet = evaluate(train_eval)
        vl, vmet = evaluate(val_loader)

        hist["epoch"].append(epoch)
        hist["train_loss"].append(tl);   hist["val_loss"].append(vl)
        for k in ("acc","prec","rec","f1","mcc"):
            hist[f"train_{k}"].append(tmet[k]); hist[f"val_{k}"].append(vmet[k])

        logger.info("Epoch %d/%d", epoch, EPOCHS)
        log_metrics("Train", tl, tmet); log_metrics("Val", vl, vmet)
        logger.info("Val CM:\n%s\n", vmet["cm"])

        if vmet["acc"] > best_acc:
            best_acc, no_imp = vmet["acc"], 0
            best_cm = vmet["cm"]
            torch.save(model.state_dict(), MODEL_OUT)
            save_cm(best_cm, "Best Confusion Matrix",
                    os.path.join(GRAPH_DIR,"swin_confmat_best.png"))
            logger.info("↑ New best %.4f saved", best_acc)
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                logger.info("Early stopping (no improve %d)", no_imp); break

        if epoch == PHASE1:
            logger.info("Unfreezing backbone & lowering LR for Phase-2 …")
            for p in model.parameters(): p.requires_grad = True
            opt = AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
            remaining = EPOCHS - PHASE1
            sched = get_cosine_schedule_with_warmup(
                opt, int(0.05*len(train_loader)*remaining),
                len(train_loader)*remaining)

    save_cm(vmet["cm"], "Final Confusion Matrix",
            os.path.join(GRAPH_DIR,"swin_confmat_final.png"))
    for k,lab in [("loss","Loss"),("acc","Accuracy"),("prec","Precision"),
                  ("rec","Recall"),("f1","F1"),("mcc","MCC")]:
        save_curve(hist, k, lab)

    pd.DataFrame(hist).round(4).to_csv(
        os.path.join(LOG_DIR,"metrics_history.csv"), index=False)

    logger.info("Training finished. Best Val Acc = %.4f", best_acc)
    print(f"Best validation accuracy: {best_acc:.4f}")

except Exception:
    logger.error("Exception:\n%s", traceback.format_exc())
    raise
