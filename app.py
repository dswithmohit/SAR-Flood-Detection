"""
SAR Flood Detection — Streamlit Web App
----------------------------------------
Frontend for the Random Forest SAR flood classifier.
Allows users to:
  1. Upload pre-processed SAR feature arrays (.npy) OR use demo data
  2. Select an AOI tile (row/col crop) via sliders
  3. Run inference and view a colour-coded flood mask overlay
  4. Download the prediction mask as a .npy or PNG
"""

import io
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import joblib
from pathlib import Path
import time

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SAR Flood Detector",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS  ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
}

h1, h2, h3 {
    font-family: 'Space Mono', monospace;
    letter-spacing: -0.5px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 12px 16px;
}
[data-testid="stMetricValue"] {
    font-family: 'Space Mono', monospace;
    color: #58a6ff !important;
    font-size: 1.6rem !important;
}
[data-testid="stMetricLabel"] {
    color: #8b949e !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Primary button */
.stButton > button {
    background: linear-gradient(135deg, #1f6feb 0%, #388bfd 100%);
    color: white;
    border: none;
    border-radius: 6px;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    padding: 0.55rem 1.4rem;
    letter-spacing: 0.5px;
    transition: opacity 0.2s;
    width: 100%;
}
.stButton > button:hover { opacity: 0.85; }

/* Info boxes */
.info-box {
    background: #161b22;
    border: 1px solid #30363d;
    border-left: 3px solid #58a6ff;
    border-radius: 6px;
    padding: 14px 18px;
    margin: 10px 0;
    font-size: 0.88rem;
    line-height: 1.6;
    color: #c9d1d9;
}

/* Badge */
.badge {
    display: inline-block;
    background: #1f6feb22;
    border: 1px solid #1f6feb55;
    color: #58a6ff;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.72rem;
    font-family: 'Space Mono', monospace;
    margin: 2px;
    letter-spacing: 0.5px;
}

/* Divider */
hr { border-color: #30363d !important; }

/* Expander */
[data-testid="stExpander"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
}

/* File uploader */
[data-testid="stFileUploadDropzone"] {
    background: #161b22 !important;
    border: 1px dashed #30363d !important;
    border-radius: 8px !important;
}

/* Selectbox / slider labels */
label { color: #8b949e !important; font-size: 0.82rem !important; text-transform: uppercase; letter-spacing: 0.8px; }

/* Plot background transparency */
.stPlot { background: transparent; }
</style>
""", unsafe_allow_html=True)


# ── helpers ───────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model(path: str):
    return joblib.load(path)


def make_demo_arrays(h=512, w=512, seed=42):
    """Generate synthetic pre/post-flood SAR-like arrays for demo purposes."""
    rng = np.random.default_rng(seed)
    pre  = rng.normal(loc=-12, scale=3, size=(h, w)).astype(np.float32)
    # Simulate flooded region: lower backscatter in a rectangular patch
    post = pre.copy()
    post[180:320, 200:380] -= rng.uniform(4, 8, size=(140, 180))
    post += rng.normal(0, 0.5, size=(h, w))
    return pre, post


def build_features(pre: np.ndarray, post: np.ndarray):
    diff = post - pre
    h, w = pre.shape
    X = np.stack([pre.ravel(), post.ravel(), diff.ravel()], axis=1)
    return X, h, w


def run_inference(model, X: np.ndarray, h: int, w: int):
    pred = model.predict(X)           # 0 = Land, 1 = Water/Flood
    mask = pred.reshape(h, w)
    proba = model.predict_proba(X)[:, 1].reshape(h, w)
    return mask, proba


def render_overlay(pre: np.ndarray, mask: np.ndarray, proba: np.ndarray):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#0d1117")

    panels = [
        (pre,   "Pre-Flood SAR (VV dB)",  "gray"),
        (mask,  "Predicted Flood Mask",    None),
        (proba, "Flood Probability",       "YlOrRd"),
    ]

    for ax, (data, title, cmap) in zip(axes, panels):
        ax.set_facecolor("#0d1117")
        if title == "Predicted Flood Mask":
            cmap_mask = mcolors.ListedColormap(["#2d333b", "#1f6feb"])
            im = ax.imshow(data, cmap=cmap_mask, vmin=0, vmax=1, interpolation="nearest")
            legend = [Patch(color="#2d333b", label="Land"),
                      Patch(color="#1f6feb", label="Water / Flood")]
            ax.legend(handles=legend, loc="lower right",
                      facecolor="#161b22", edgecolor="#30363d",
                      labelcolor="#e6edf3", fontsize=8)
        else:
            im = ax.imshow(data, cmap=cmap, interpolation="bilinear")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                         label="dB" if "Pre" in title else "P(flood)")

        ax.set_title(title, fontfamily="monospace", fontsize=10,
                     color="#e6edf3", pad=8)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    plt.tight_layout(pad=1.5)
    return fig


def fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="#0d1117")
    buf.seek(0)
    return buf.read()


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🛰️ SAR Flood Detector")
    st.markdown('<span class="badge">RF Classifier</span>'
                '<span class="badge">Sentinel-1</span>'
                '<span class="badge">VV Polarisation</span>',
                unsafe_allow_html=True)
    st.markdown("---")

    # ── Model ──
    st.markdown("#### Model")
    model_path_default = "models/sar_flood_rf_model.pkl"
    model_path = st.text_input("Model path (.pkl)", value=model_path_default,
                                label_visibility="collapsed",
                                placeholder="models/sar_flood_rf_model.pkl")

    model = None
    model_loaded = False
    if Path(model_path).exists():
        with st.spinner("Loading model…"):
            model = load_model(model_path)
        model_loaded = True
        st.success("✓ Model loaded", icon="✅")
    else:
        st.warning("Model file not found — demo predictions will be shown.", icon="⚠️")

    st.markdown("---")

    # ── Data source ──
    st.markdown("#### Data Source")
    data_mode = st.radio("", ["Use Demo Data", "Upload .npy Arrays"],
                         label_visibility="collapsed")

    pre_arr = post_arr = None

    if data_mode == "Upload .npy Arrays":
        st.caption("Upload pre- and post-flood VV backscatter arrays (2-D float32).")
        pre_file  = st.file_uploader("Pre-flood array",  type=["npy"], key="pre")
        post_file = st.file_uploader("Post-flood array", type=["npy"], key="post")
        if pre_file and post_file:
            pre_arr  = np.load(pre_file).astype(np.float32)
            post_arr = np.load(post_file).astype(np.float32)
            if pre_arr.ndim != 2 or pre_arr.shape != post_arr.shape:
                st.error("Arrays must be 2-D and the same shape.")
                pre_arr = post_arr = None
    else:
        st.caption("Synthetic 512×512 SAR scene with a simulated flood patch.")
        demo_seed = st.slider("Scene seed", 1, 99, 42)
        pre_arr, post_arr = make_demo_arrays(seed=demo_seed)

    st.markdown("---")

    # ── AOI crop ──
    st.markdown("#### AOI — Tile Selection")
    if pre_arr is not None:
        H, W = pre_arr.shape
        aoi_size = st.slider("Tile size (px)", 64, min(H, W, 512), 256, step=64)
        row_start = st.slider("Row offset", 0, max(0, H - aoi_size), 0)
        col_start = st.slider("Col offset", 0, max(0, W - aoi_size), 0)
    else:
        aoi_size = row_start = col_start = None

    st.markdown("---")
    run_btn = st.button("⚡ Run Inference")


# ── main area ─────────────────────────────────────────────────────────────────

st.markdown("# SAR Flood Detection")
st.markdown(
    "Random Forest classifier trained on **Sentinel-1 VV** backscatter. "
    "Select an AOI tile in the sidebar, then hit **Run Inference**.",
    unsafe_allow_html=False,
)
st.markdown("---")

if pre_arr is None:
    st.markdown(
        '<div class="info-box">👈 Configure a data source in the sidebar to get started.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

# ── quick scene stats ─────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Scene height", f"{pre_arr.shape[0]} px")
col2.metric("Scene width",  f"{pre_arr.shape[1]} px")
col3.metric("Pre-flood mean (dB)", f"{pre_arr.mean():.1f}")
col4.metric("Post-flood mean (dB)", f"{post_arr.mean():.1f}")

st.markdown("---")

# ── AOI preview ───────────────────────────────────────────────────────────────
aoi_pre  = pre_arr [row_start:row_start+aoi_size, col_start:col_start+aoi_size]
aoi_post = post_arr[row_start:row_start+aoi_size, col_start:col_start+aoi_size]

with st.expander("📡 AOI Preview  (pre vs post flood)", expanded=True):
    fig_prev, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), facecolor="#0d1117")
    for ax, data, title in [(ax1, aoi_pre, "Pre-Flood VV"), (ax2, aoi_post, "Post-Flood VV")]:
        ax.imshow(data, cmap="gray", interpolation="bilinear")
        ax.set_title(title, fontfamily="monospace", fontsize=9, color="#e6edf3", pad=6)
        ax.axis("off")
        ax.set_facecolor("#0d1117")
    plt.tight_layout(pad=1)
    st.pyplot(fig_prev, use_container_width=True)
    plt.close(fig_prev)

# ── inference ─────────────────────────────────────────────────────────────────
if run_btn:
    with st.spinner("Running inference…"):
        t0 = time.time()
        X, h, w = build_features(aoi_pre, aoi_post)

        if model_loaded:
            mask, proba = run_inference(model, X, h, w)
        else:
            # Demo fallback: threshold the difference image
            diff = aoi_post - aoi_pre
            proba = (diff - diff.min()) / (diff.max() - diff.min() + 1e-9)
            proba = 1.0 - proba          # lower backscatter → higher flood prob
            mask  = (proba > 0.6).astype(np.uint8)

        elapsed = time.time() - t0

    st.success(f"Inference complete in **{elapsed*1000:.0f} ms**", icon="✅")
    st.markdown("---")

    # ── results metrics ───────────────────────────────────────────────────────
    flood_pct = mask.mean() * 100
    land_pct  = 100 - flood_pct
    c1, c2, c3 = st.columns(3)
    c1.metric("Flood / Water", f"{flood_pct:.1f}%")
    c2.metric("Land",          f"{land_pct:.1f}%")
    c3.metric("Tile size",     f"{h}×{w} px")

    st.markdown("---")

    # ── overlay plot ──────────────────────────────────────────────────────────
    st.markdown("#### Results")
    fig_out = render_overlay(aoi_pre, mask, proba)
    st.pyplot(fig_out, use_container_width=True)

    # ── downloads ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Download")
    dl1, dl2 = st.columns(2)

    png_bytes = fig_to_bytes(fig_out)
    dl1.download_button(
        "⬇ Download overlay PNG",
        data=png_bytes,
        file_name="sar_flood_result.png",
        mime="image/png",
    )

    mask_buf = io.BytesIO()
    np.save(mask_buf, mask)
    dl2.download_button(
        "⬇ Download mask (.npy)",
        data=mask_buf.getvalue(),
        file_name="flood_mask.npy",
        mime="application/octet-stream",
    )

    plt.close(fig_out)

# ── footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center; color:#484f58; font-size:0.78rem; '
    'font-family:\'Space Mono\',monospace; padding: 8px 0;">'
    'SAR Flood Detector · Sentinel-1 VV · Random Forest · Assignment Demo'
    '</div>',
    unsafe_allow_html=True,
)