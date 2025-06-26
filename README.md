# Multimodal-Hateful-meme-detection_Updated

Multimodal Hateful Meme Detection using Knowledge Distillation and Hierarchical Vision Transformer Framework
This project presents a multimodal deep learning framework that detects hateful content in memes by fusing visual and textual features. It introduces an efficient pipeline that leverages Knowledge Distillation for textual feature compression and a Hierarchical Vision Transformer (HVT) for visual understanding, enabling accurate and scalable hate speech detection in complex multimodal inputs.

Textual Branch – Knowledge Distillation
Uses a distilled BERT model to extract semantic-rich text embeddings efficiently.

Knowledge distillation compresses a large teacher model into a smaller, faster student model without significant performance loss.

Efficient for real-time or edge deployment (e.g., mobile devices).

🖼️ Visual Branch – Hierarchical Vision Transformer (HVT)
Employs a Hierarchical Vision Transformer for multi-scale visual understanding.

Captures both local and global dependencies using a window-based attention mechanism.

Reduces computational complexity through sliding window attention and hierarchical token merging.

🔄 Cross-Modal Fusion
Extracted text and image features are concatenated to form a fused representation F.

F is passed through a softmax classifier for final label prediction.

📚** Datasets Used**
MMHS150K
Twitter-based multimodal hate speech dataset with 150K samples.

Hateful Memes Challenge (HMC)
Curated by Facebook AI to emphasize the need for multimodal reasoning.

MultiOFF
A small-scale dataset of offensive memes with multimodal annotations.

Training Overview
Uses standard supervised learning with cross-entropy loss.

Employs adaptive learning rate decay, early stopping, and batch-based training.

Visual features processed through HVT layers and pooled before fusion.

Textual features extracted via a distilled transformer with tokenization and embedding layers.

**Experimental Setup**
The experiments were conducted across two different hardware configurations to ensure robustness, performance benchmarking, and scalability of the proposed model.

🔧 Primary Setup: NVIDIA TITAN RTX GPUs
For large-scale training (e.g., MMHS150K and Hateful Memes Challenge), experiments were conducted using a high-performance deep learning workstation equipped with:

Component	Details
GPU	2 × NVIDIA TITAN RTX (24 GB each, operated in parallel)
Total VRAM	48 GB
Framework	PyTorch with CUDA & cuDNN
Usage	Parallel GPU training for multimodal feature extraction and fusion

This configuration was used for accelerated training of transformer-based models with large batch sizes and high-resolution image inputs.

**Import Libraries**
 "import os\n",
    "import pandas as pd\n",
    "\n",
    "import torch\n",
    "from torch.nn.utils.rnn import pad_sequence\n",
    "from torch.utils.data import DataLoader, Dataset\n",
    "from PIL import Image\n",
    "import torch.nn as nn\n",
    "import torch.optim as optim\n",
    "import torchvision.models as models\n",
    "from datetime import datetime\n",
    "import torch.optim.lr_scheduler as scheduler\n",
    "import json\n",
    "import torchvision.transforms as transforms\n",
    "import random\n",
    "\n",
    "from transformers import DistilBertForSequenceClassification\n",
    "from transformers import DistilBertTokenizerFast\n",
    "from torchvision import transforms\n",
    "from transformers import get_cosine_schedule_with_warmup\n",
    "from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             confusion_matrix, precision_score, recall_score)"
    "import timm\n",
    "from timm.loss import SoftTargetCrossEntropy\n",
    "from timm.data import Mixup"


