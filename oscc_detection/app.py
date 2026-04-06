#!/usr/bin/env python3
"""
============================================================
OSCC Detection — Streamlit Demo App
============================================================
Microscope-Aided Deep Learning Framework for Early Oral
Cancer (OSCC) Detection

Upload a histopathology image → get instant prediction,
risk stage, confidence score, and Grad-CAM explanation.

Run with:
    streamlit run app.py
============================================================
"""

import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── project paths ─────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ── suppress TF warnings ──────────────────────────────────
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# ══════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="OSCC Detection System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════
#  CUSTOM CSS — dark clinical theme
# ══════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
}

.main { background-color: #0d1117; }

/* Header */
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #58a6ff 0%, #79c0ff 50%, #a5d6ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
    line-height: 1.2;
}

.hero-sub {
    font-size: 1.0rem;
    color: #8b949e;
    font-weight: 300;
    margin-bottom: 2rem;
    letter-spacing: 0.05em;
}

/* Result cards */
.result-card {
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 0.5rem 0;
    text-align: center;
}

.result-card-danger {
    border-color: #f85149;
    background: linear-gradient(135deg, #1a0d0d 0%, #200d0d 100%);
}

.result-card-safe {
    border-color: #3fb950;
    background: linear-gradient(135deg, #0d1a0d 0%, #0d200d 100%);
}

.result-card-warning {
    border-color: #d29922;
    background: linear-gradient(135deg, #1a160d 0%, #201a0d 100%);
}

.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2.8rem;
    font-weight: 700;
    line-height: 1;
    margin: 0.5rem 0;
}

.metric-label {
    font-size: 0.75rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #8b949e;
    margin-bottom: 0.3rem;
}

.metric-value-danger  { color: #f85149; }
.metric-value-safe    { color: #3fb950; }
.metric-value-warning { color: #d29922; }
.metric-value-info    { color: #58a6ff; }

/* Stage badge */
.stage-badge {
    display: inline-block;
    padding: 0.4rem 1.2rem;
    border-radius: 999px;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    margin-top: 0.5rem;
}

.stage-normal   { background: #0d3d1a; color: #3fb950; border: 1px solid #3fb950; }
.stage-mild     { background: #2d1f00; color: #d29922; border: 1px solid #d29922; }
.stage-moderate { background: #2d1500; color: #f0883e; border: 1px solid #f0883e; }
.stage-severe   { background: #2d0000; color: #f85149; border: 1px solid #f85149; }

/* Info box */
.info-box {
    background: #161b22;
    border: 1px solid #21262d;
    border-left: 4px solid #58a6ff;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 1rem 0;
    font-size: 0.9rem;
    color: #c9d1d9;
}

/* Upload area */
.upload-hint {
    background: #161b22;
    border: 2px dashed #30363d;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    color: #8b949e;
    font-size: 0.9rem;
}

/* Section headers */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #58a6ff;
    border-bottom: 1px solid #21262d;
    padding-bottom: 0.5rem;
    margin: 1.5rem 0 1rem 0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #21262d;
}

/* Streamlit overrides */
.stButton > button {
    background: linear-gradient(135deg, #1f6feb 0%, #388bfd 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 2rem;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    font-size: 0.95rem;
    width: 100%;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

div[data-testid="stFileUploader"] {
    border-radius: 12px;
}

.disclaimer {
    background: #161b22;
    border: 1px solid #d29922;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-size: 0.8rem;
    color: #d29922;
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  MODEL LOADING (cached)
# ══════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_model():
    """Load full OSCC model with trained weights."""
    try:
        from model import build_oscc_model, build_lite_model

        config_path = PROJECT_ROOT / "outputs" / "models" / "model_config.json"
        weights_path = PROJECT_ROOT / "outputs" / "models" / "best_model.weights.h5"

        config = {}
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)

        # Load lite model with trained weights
        lite_model = build_lite_model(config)
        if weights_path.exists():
            lite_model.load_weights(str(weights_path))

        # Build full model and transfer weights
        full_model = build_oscc_model(config)
        lite_weights = lite_model.weights
        full_lite_weights = full_model.lite.weights

        transferred = 0
        for lw, fw in zip(lite_weights, full_lite_weights):
            if lw.shape == fw.shape:
                fw.assign(lw)
                transferred += 1

        return full_model, transferred, True

    except Exception as e:
        return None, 0, str(e)


@st.cache_resource(show_spinner=False)
def load_patch_extractor():
    from patch_extractor import PatchExtractor
    return PatchExtractor(patch_size=224, min_image_size=448)


# ══════════════════════════════════════════════════════════
#  STAGING LOGIC
# ══════════════════════════════════════════════════════════
def get_stage(oscc_prob: float, pred_class: str):
    """
    Map OSCC probability to clinical-style risk stage.
    Returns (stage_name, stage_class, description, recommendation)
    """
    if pred_class == "normal":
        if oscc_prob < 0.15:
            return ("Normal Tissue", "stage-normal",
                    "No signs of malignancy detected.",
                    "Routine follow-up recommended.")
        else:
            return ("Low Suspicion", "stage-mild",
                    "Tissue appears normal but borderline features present.",
                    "Clinical correlation advised.")
    else:
        if oscc_prob < 0.65:
            return ("Mild Dysplasia", "stage-mild",
                    "Early cellular abnormalities detected.",
                    "Biopsy and specialist review recommended.")
        elif oscc_prob < 0.80:
            return ("Moderate Dysplasia", "stage-moderate",
                    "Significant cellular changes consistent with pre-malignancy.",
                    "Urgent specialist referral required.")
        else:
            return ("Severe / OSCC", "stage-severe",
                    "High-confidence OSCC features detected.",
                    "Immediate oncological consultation required.")


# ══════════════════════════════════════════════════════════
#  GRAD-CAM
# ══════════════════════════════════════════════════════════
def compute_gradcam(model, image_patch: np.ndarray) -> np.ndarray:
    """Compute Grad-CAM heatmap for a single patch."""
    try:
        # Find backbone
        backbone = None
        for layer in model.layers:
            if hasattr(layer, "layers"):
                for sl in layer.layers:
                    if isinstance(sl, tf.keras.layers.Conv2D):
                        backbone = layer
                        break
            if backbone:
                break

        if backbone is None:
            return np.zeros((224, 224), dtype=np.float32)

        # Find last conv layer
        last_conv = None
        for layer in backbone.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv = layer.name

        if last_conv is None:
            return np.zeros((224, 224), dtype=np.float32)

        grad_model = tf.keras.Model(
            inputs=backbone.input,
            outputs=[backbone.get_layer(last_conv).output, backbone.output],
        )

        img_tensor = tf.convert_to_tensor(image_patch[np.newaxis], dtype=tf.float32)
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(img_tensor)
            target = preds[:, 1] if preds.shape[-1] == 2 else tf.reduce_mean(preds)

        grads = tape.gradient(target, conv_out)
        if grads is None:
            return np.zeros((224, 224), dtype=np.float32)

        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.reduce_sum(conv_out[0] * pooled, axis=-1).numpy()
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
        return cv2.resize(heatmap, (224, 224)).astype(np.float32)

    except Exception:
        return np.zeros((224, 224), dtype=np.float32)


def make_overlay(image: np.ndarray, heatmap: np.ndarray, alpha=0.45) -> np.ndarray:
    if image.dtype != np.uint8:
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    h_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    h_color = cv2.cvtColor(h_color, cv2.COLOR_BGR2RGB)
    if h_color.shape[:2] != image.shape[:2]:
        h_color = cv2.resize(h_color, (image.shape[1], image.shape[0]))
    return cv2.addWeighted(image, 1 - alpha, h_color, alpha, 0)


# ══════════════════════════════════════════════════════════
#  PREDICT
# ══════════════════════════════════════════════════════════
def run_prediction(model, patch_extractor, image_path: str):
    """Run full pipeline: extract patches → predict → GradCAM."""
    patches, orig_shape = patch_extractor.extract_from_file(image_path)
    num_patches = 4
    while len(patches) < num_patches:
        patches.append(patches[len(patches) % len(patches)])
    patches = patches[:num_patches]

    patch_array = np.stack(patches, axis=0)[np.newaxis].astype(np.float32)
    output = model(patch_array, training=False)
    preds = output["predictions"].numpy()[0]
    attn_weights = output["attention_weights"].numpy()[0].flatten()

    pred_class = "oscc" if np.argmax(preds) == 1 else "normal"
    confidence = float(np.max(preds))
    oscc_prob = float(preds[1])
    normal_prob = float(preds[0])

    # GradCAM on most attended patch
    best_idx = int(np.argmax(attn_weights))
    heatmap = compute_gradcam(model, patches[best_idx])
    overlay = make_overlay(patches[best_idx], heatmap)

    return {
        "pred_class": pred_class,
        "confidence": confidence,
        "oscc_prob": oscc_prob,
        "normal_prob": normal_prob,
        "attn_weights": attn_weights,
        "patches": patches,
        "best_patch_idx": best_idx,
        "heatmap": heatmap,
        "overlay": overlay,
        "orig_shape": orig_shape,
    }


# ══════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🔬 OSCC Detection")
    st.markdown("---")
    st.markdown("""
**About this system**

This tool uses a deep learning pipeline to analyse oral histopathology images for signs of Oral Squamous Cell Carcinoma (OSCC).

**Pipeline:**
- EfficientNetB0 (feature extraction)
- Positional Encoding
- Soft Attention
- GRU Context Learner
- MLP Classifier

**Performance:**
- Test Accuracy: **91%**
- AUC-ROC: **0.975**
""")
    st.markdown("---")
    st.markdown("""
**Staging Guide**

🟢 **Normal** — No malignancy  
🟡 **Mild Dysplasia** — Early changes  
🟠 **Moderate Dysplasia** — Pre-malignant  
🔴 **Severe / OSCC** — Malignant features
""")
    st.markdown("---")
    st.caption("Final Year Project | OSCC Detection Team | 2026")


# ══════════════════════════════════════════════════════════
#  MAIN LAYOUT
# ══════════════════════════════════════════════════════════
st.markdown('<div class="hero-title">🔬 OSCC Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Microscope-Aided Deep Learning Framework for Early Oral Cancer Detection</div>', unsafe_allow_html=True)

# Load model
with st.spinner("Loading AI model..."):
    model, transferred, status = load_model()
    patch_extractor = load_patch_extractor()

if model is None:
    st.error(f"❌ Model failed to load: {status}")
    st.info("Make sure you have run `train.py` first and weights exist at `outputs/models/best_model.weights.h5`")
    st.stop()
else:
    st.success(f"✅ Model loaded — {transferred}/28 weight tensors active", icon="🧠")

st.markdown("---")

# Upload section
st.markdown('<div class="section-header">📤 Upload Histopathology Image</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload a histopathology slide image (JPG, PNG)",
    type=["jpg", "jpeg", "png", "bmp", "tiff"],
    help="Upload a microscopy image from the OSCC dataset or your own sample.",
)

if uploaded_file is not None:
    # Save to temp file
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    col_img, col_results = st.columns([1, 1], gap="large")

    with col_img:
        st.markdown('<div class="section-header">🖼️ Uploaded Image</div>', unsafe_allow_html=True)
        st.image(tmp_path, use_container_width=True, caption=uploaded_file.name)

    # Run prediction
    with st.spinner("🧠 Analysing image..."):
        try:
            result = run_prediction(model, patch_extractor, tmp_path)
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            st.stop()

    stage_name, stage_class, stage_desc, stage_rec = get_stage(
        result["oscc_prob"], result["pred_class"]
    )

    with col_results:
        st.markdown('<div class="section-header">📊 Prediction Results</div>', unsafe_allow_html=True)

        # Prediction card
        is_oscc = result["pred_class"] == "oscc"
        card_class = "result-card-danger" if is_oscc else "result-card-safe"
        val_class = "metric-value-danger" if is_oscc else "metric-value-safe"
        pred_label = "OSCC DETECTED" if is_oscc else "NORMAL TISSUE"
        pred_emoji = "⚠️" if is_oscc else "✅"

        st.markdown(f"""
        <div class="result-card {card_class}">
            <div class="metric-label">Diagnosis</div>
            <div class="metric-value {val_class}">{pred_emoji} {pred_label}</div>
            <div class="stage-badge {stage_class}">{stage_name}</div>
        </div>
        """, unsafe_allow_html=True)

        # Metrics row
        c1, c2 = st.columns(2)
        with c1:
            conf_pct = f"{result['confidence']*100:.1f}%"
            st.markdown(f"""
            <div class="result-card">
                <div class="metric-label">Confidence</div>
                <div class="metric-value metric-value-info">{conf_pct}</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            oscc_pct = f"{result['oscc_prob']*100:.1f}%"
            risk_class = "metric-value-danger" if result["oscc_prob"] > 0.5 else "metric-value-safe"
            st.markdown(f"""
            <div class="result-card">
                <div class="metric-label">OSCC Probability</div>
                <div class="metric-value {risk_class}">{oscc_pct}</div>
            </div>
            """, unsafe_allow_html=True)

        # Stage info
        st.markdown(f"""
        <div class="info-box">
            <strong>🩺 Clinical Assessment:</strong> {stage_desc}<br><br>
            <strong>📋 Recommendation:</strong> {stage_rec}
        </div>
        """, unsafe_allow_html=True)

        # Probability bar
        st.markdown('<div class="metric-label" style="margin-top:1rem">Class Probabilities</div>', unsafe_allow_html=True)
        prob_col1, prob_col2 = st.columns(2)
        with prob_col1:
            st.metric("Normal", f"{result['normal_prob']*100:.1f}%")
        with prob_col2:
            st.metric("OSCC", f"{result['oscc_prob']*100:.1f}%")

        st.progress(result["oscc_prob"], text="OSCC Risk")

    # ── XAI Section ───────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">🔍 Explainability — Grad-CAM + Attention</div>', unsafe_allow_html=True)

    xcol1, xcol2, xcol3 = st.columns(3)

    best = result["best_patch_idx"]
    best_patch_disp = result["patches"][best]
    if best_patch_disp.max() <= 1.0:
        best_patch_disp = (best_patch_disp * 255).astype(np.uint8)

    with xcol1:
        st.image(best_patch_disp, caption=f"Most Attended Patch (#{best})", use_container_width=True)

    with xcol2:
        heatmap_color = cv2.applyColorMap(
            (result["heatmap"] * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        st.image(heatmap_color, caption="Grad-CAM Heatmap", use_container_width=True)

    with xcol3:
        st.image(result["overlay"], caption="Grad-CAM Overlay", use_container_width=True)

    # Attention weights chart
    st.markdown('<div class="section-header">📈 Patch Attention Weights</div>', unsafe_allow_html=True)
    attn = result["attn_weights"]
    fig, ax = plt.subplots(figsize=(8, 2.5), facecolor="#161b22")
    ax.set_facecolor("#0d1117")
    colors = ["#f85149" if i == result["best_patch_idx"] else "#388bfd" for i in range(len(attn))]
    bars = ax.bar([f"Patch {i}" for i in range(len(attn))], attn, color=colors, width=0.5)
    ax.set_ylabel("Attention Weight", color="#8b949e", fontsize=10)
    ax.set_title("Model Attention per Patch  (red = highest)", color="#c9d1d9", fontsize=11)
    ax.tick_params(colors="#8b949e")
    ax.spines["bottom"].set_color("#30363d")
    ax.spines["left"].set_color("#30363d")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, val in zip(bars, attn):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", color="#e6edf3", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
    ⚠️ <strong>Disclaimer:</strong> This tool is intended for educational and research demonstration purposes only.
    It is not a certified medical device and must not be used for clinical diagnosis.
    Always consult a qualified medical professional for diagnosis and treatment decisions.
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="upload-hint">
        <div style="font-size:3rem;margin-bottom:1rem">🔬</div>
        <div style="font-size:1.1rem;color:#c9d1d9;margin-bottom:0.5rem">Upload a histopathology image to begin analysis</div>
        <div style="font-size:0.85rem">Supported formats: JPG, PNG, BMP, TIFF</div>
        <div style="font-size:0.85rem;margin-top:0.5rem">Best results with H&E stained microscopy slides at 10x–40x magnification</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">ℹ️ How It Works</div>', unsafe_allow_html=True)
    h1, h2, h3, h4 = st.columns(4)
    for col, icon, title, desc in [
        (h1, "📤", "Upload", "Upload any H&E stained oral tissue microscopy image"),
        (h2, "🧠", "Analyse", "EfficientNetB0 extracts features, GRU models context"),
        (h3, "🎯", "Predict", "Soft attention + MLP classifies normal vs OSCC"),
        (h4, "🔍", "Explain", "Grad-CAM highlights which regions drove the decision"),
    ]:
        with col:
            st.markdown(f"""
            <div class="result-card" style="padding:1.2rem">
                <div style="font-size:2rem">{icon}</div>
                <div style="font-weight:600;margin:0.5rem 0;color:#c9d1d9">{title}</div>
                <div style="font-size:0.82rem;color:#8b949e">{desc}</div>
            </div>
            """, unsafe_allow_html=True)