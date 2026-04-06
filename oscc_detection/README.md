# 🔬 OSCC Detection — Deep Learning Framework for Oral Cancer Diagnosis

**Microscope-Aided Deep Learning Framework for Early Oral Squamous Cell Carcinoma (OSCC) Detection**

A complete deep learning pipeline that analyses histopathology microscope images of oral tissue to classify them as **Normal** or **OSCC (Cancerous)**, with clinical risk levels and Grad-CAM visual explanations.

## ✨ Key Features

- **Automated dataset download** from Kaggle with train/val/test splitting
- **Patch-based analysis** — images split into 224×224 patches for fine-grained inspection
- **EfficientNetB0 backbone** (pretrained on ImageNet) for deep feature extraction
- **Soft attention mechanism** — learns which patches are diagnostically important
- **Bidirectional LSTM** — captures sequential context between patches
- **Clinical risk levels** — Low / Medium / High based on OSCC probability
- **Grad-CAM heatmaps** — visual explanation of model predictions
- **Interactive demo notebook** for presentations and reviews

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                    OSCC Detection Pipeline                        │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Input Image (Histopathology Slide)                               │
│       │                                                            │
│       ▼                                                            │
│  ┌──────────────────┐                                              │
│  │  Patch Extractor  │  Split into 224×224 patches                 │
│  └────────┬─────────┘                                              │
│           │  (num_patches, 224, 224, 3)                             │
│           ▼                                                        │
│  ┌──────────────────┐                                              │
│  │  EfficientNetB0   │  Pretrained CNN backbone                    │
│  │  + GAP + Dense    │  Extract 1280-dim feature per patch         │
│  └────────┬─────────┘                                              │
│           │  (num_patches, 1280)                                   │
│           ▼                                                        │
│  ┌──────────────────┐                                              │
│  │ Positional Enc.   │  Sinusoidal spatial encoding                │
│  └────────┬─────────┘                                              │
│           │                                                        │
│     ┌─────┴─────┐                                                  │
│     ▼           ▼                                                  │
│  ┌────────┐  ┌──────────────┐                                      │
│  │  Soft   │  │ Bidirectional │                                     │
│  │Attention│  │   LSTM (×2)   │                                     │
│  └────┬───┘  └──────┬───────┘                                      │
│       │ (1280)      │ (512)                                        │
│       └──────┬──────┘                                              │
│              │  Concatenate → Dense(512)                            │
│              ▼                                                      │
│  ┌──────────────────┐                                              │
│  │  MLP Classifier   │  Dense(512) → Dense(256) → Dense(2)        │
│  └────────┬─────────┘                                              │
│           │                                                        │
│           ▼                                                        │
│  ┌──────────────────┐                                              │
│  │  Output:          │                                              │
│  │  • Normal / OSCC  │  ← Class prediction                        │
│  │  • Confidence %   │  ← Softmax probability                     │
│  │  • Risk Level     │  ← Low / Medium / High                     │
│  │  • Grad-CAM       │  ← Heatmap explanation                     │
│  └──────────────────┘                                              │
└────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
oscc_detection/
├── data/                           ← Dataset (auto-downloaded)
│   ├── raw/                        ← Extracted zip contents
│   ├── train/normal/, train/oscc/  ← Training set (70%)
│   ├── val/normal/, val/oscc/      ← Validation set (15%)
│   └── test/normal/, test/oscc/    ← Test set (15%)
├── src/
│   ├── download_dataset.py         ← Kaggle download + split
│   ├── preprocess.py               ← Data generator + augmentation
│   ├── patch_extractor.py          ← Image → patches
│   ├── cnn_feature_extractor.py    ← EfficientNetB0 / ResNet50 backbone
│   ├── attention_module.py         ← Soft + multi-head attention
│   ├── lstm_context.py             ← BiLSTM / BiGRU context learner
│   ├── mlp_classifier.py           ← Classification head + risk levels
│   ├── model.py                    ← Integrated end-to-end model
│   ├── train.py                    ← Training pipeline + callbacks
│   ├── evaluate.py                 ← Metrics + confusion matrix + ROC
│   └── gradcam.py                  ← Grad-CAM + attention visualisation
├── outputs/
│   ├── models/                     ← Saved model weights + config
│   ├── plots/                      ← Learning curves, confusion matrix
│   └── heatmaps/                   ← Grad-CAM explanation images
├── notebooks/
│   └── demo.ipynb                  ← Interactive demo notebook
├── requirements.txt
└── README.md
```

---

## 📋 Requirements

- **Python** 3.9+
- **GPU** recommended (NVIDIA with CUDA) — CPU is supported but slower
- ~2 GB disk space for dataset + model weights

### Key Packages

| Package | Version | Purpose |
|---------|---------|---------|
| TensorFlow | ≥ 2.12 | Deep learning framework |
| OpenCV | ≥ 4.8 | Image processing |
| scikit-learn | ≥ 1.3 | Metrics and evaluation |
| matplotlib / seaborn | ≥ 3.7 | Visualisation |
| kaggle | ≥ 1.5 | Dataset download |
| tf-keras-vis | ≥ 0.8 | Grad-CAM utilities |

---

## 🚀 Installation

```bash
# 1. Clone the repository
git clone <repository-url>
cd oscc_detection

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 📥 Dataset Setup

1. **Get a Kaggle API key:**
   - Go to [kaggle.com/account](https://www.kaggle.com/account)
   - Scroll to **API** → Click **Create New Token**
   - Save `kaggle.json` to `~/.kaggle/` (Linux/macOS) or `C:\Users\<You>\.kaggle\` (Windows)

2. **Download and organise the dataset:**
   ```bash
   python src/download_dataset.py
   ```

3. **Expected output:**
   ```
   ✅ Kaggle credentials found for user: your_username
   ⏳ Downloading dataset …
   ✅ Download complete.
   ⏳ Splitting into train / val / test (70 / 15 / 15) …

   =====================================================
              📊  DATASET STATISTICS
   =====================================================
   Split        Normal       OSCC      Total
   -----------------------------------------------
   train          XXX        XXX        XXX
   val            XXX        XXX        XXX
   test           XXX        XXX        XXX
   =====================================================
   ```

---

## 🏋️ Training

```bash
python src/train.py --epochs 50 --backbone efficientnet --rnn_type lstm
```

### Command-Line Arguments

| Argument | Default | Options | Description |
|----------|---------|---------|-------------|
| `--epochs` | 50 | int | Maximum training epochs |
| `--batch_size` | 16 | int | Batch size |
| `--backbone` | efficientnet | efficientnet, resnet | CNN backbone |
| `--rnn_type` | lstm | lstm, gru | Sequential context module |
| `--attention` | soft | soft, multihead | Attention mechanism |
| `--lr` | 0.001 | float | Initial learning rate |
| `--num_patches` | 4 | int | Patches per image |

Training automatically saves:
- Best model weights → `outputs/models/best_model.weights.h5`
- Learning curves → `outputs/plots/`
- Training log → `outputs/training_log.csv`

---

## 📊 Evaluation

```bash
python src/evaluate.py
```

Generates:
- Classification report (per-class precision, recall, F1)
- Confusion matrix → `outputs/plots/confusion_matrix.png`
- ROC curve → `outputs/plots/roc_curve.png`
- Metrics CSV → `outputs/metrics.csv`

---

## 🔍 Explainability (Grad-CAM)

```bash
python src/gradcam.py
```

Generates 4-panel explanation images for test samples showing:
1. **Original image**
2. **Attention weight map** — which patches the model considers important
3. **Grad-CAM heatmap** — CNN activation regions
4. **Overlay** — combined visualisation

Saved to `outputs/heatmaps/`.

---

## 📓 Demo Notebook

```bash
cd notebooks
jupyter notebook demo.ipynb
```

The notebook walks through the complete pipeline interactively:
load image → extract patches → predict → explain with Grad-CAM.

---

## 📈 Results (Placeholder)

| Metric | Value |
|--------|-------|
| Accuracy | XX.X% |
| Precision (macro) | XX.X% |
| Recall (macro) | XX.X% |
| F1 Score (macro) | XX.X% |
| AUC-ROC | X.XXX |

> *Values will be populated after training.*

---

## 🔮 Future Work

- **Multi-centre validation** — test on datasets from multiple hospitals
- **Multi-class grading** — extend to well / moderately / poorly differentiated OSCC
- **Whole-slide image (WSI) support** — process full diagnostic slides
- **Multimodal integration** — combine with clinical metadata
- **Hospital system deployment** — DICOM integration and PACS connectivity
- **Mobile inference** — TFLite model for point-of-care screening

---

## 📚 References

1. **Dataset:** Ashenafi Fasil Kebede, *Oral Cancer (OSCC) Histopathology Image Dataset*, Kaggle.
   https://www.kaggle.com/datasets/ashenafifasilkebede/dataset
2. **EfficientNet:** Tan, M. & Le, Q. V. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* ICML.
3. **Grad-CAM:** Selvaraju, R. R. et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV.
4. **Attention Mechanism:** Bahdanau, D., Cho, K. & Bengio, Y. (2015). *Neural Machine Translation by Jointly Learning to Align and Translate.* ICLR.

---

## 📄 License

This project is for educational and research purposes (Final Year Project).
