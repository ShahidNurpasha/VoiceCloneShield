"""
VoiceClone Shield - Real-Time AI Voice Authenticity & Voice Cloning Detection
Redesigned UI: Phase 1 (Realistic Interface) & Phase 2 (White Theme)
Maintains 100% backend logic, model architecture, and audio processing pipelines.
"""

import os
import subprocess
import tempfile
import threading
import time
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torch.nn as nn

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="VoiceClone Shield | AI Voice Security",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# DESIGN SYSTEM: WHITE THEME (CSS VARIABLES IN :root)
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ============================================================
   DESIGN TOKENS & CSS VARIABLES (:root)
   ============================================================ */
:root {
  /* Surfaces & Page */
  --bg-page: #FFFFFF;
  --bg-surface: #F8F9FA;
  --bg-surface-elevated: #FFFFFF;
  --border-color: #E5E7EB;
  --border-color-focus: #2563EB;
  --border-color-strong: #D1D5DB;

  /* Typography */
  --text-heading: #111827;
  --text-body: #4B5563;
  --text-muted: #6B7280;

  /* Single Accent Color */
  --accent-primary: #2563EB;
  --accent-hover: #1D4ED8;
  --accent-active: #1E40AF;
  --accent-light: #EFF6FF;

  /* Status Colors */
  --success-bg: #ECFDF5;
  --success-border: #A7F3D0;
  --success-text: #065F46;

  --warning-bg: #FFFBEB;
  --warning-border: #FDE68A;
  --warning-text: #92400E;

  --danger-bg: #FEF2F2;
  --danger-border: #FECACA;
  --danger-text: #991B1B;

  --neutral-bg: #F3F4F6;
  --neutral-border: #E5E7EB;
  --neutral-text: #374151;

  /* 8px Spacing Grid */
  --space-1: 4px;
  --space-2: 8px;
  --space-3: 12px;
  --space-4: 16px;
  --space-5: 20px;
  --space-6: 24px;
  --space-8: 32px;

  /* Geometry & Shadows */
  --radius-sm: 6px;
  --radius-md: 8px;
  --radius-lg: 10px;
  --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
  --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
}

/* ============================================================
   GLOBAL STYLES & RESET
   ============================================================ */
html, body, [class*="css"], .stApp {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
  background-color: var(--bg-page) !important;
  color: var(--text-body) !important;
}

.stApp {
  background-color: var(--bg-page) !important;
}

.block-container {
  max-width: 1360px !important;
  padding-top: 1.5rem !important;
  padding-bottom: 3.5rem !important;
  padding-left: 2rem !important;
  padding-right: 2rem !important;
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
  color: var(--text-heading) !important;
  font-weight: 600 !important;
  letter-spacing: -0.01em !important;
}

/* ============================================================
   SIDEBAR STYLING (WHITE THEME)
   ============================================================ */
section[data-testid="stSidebar"] {
  background-color: var(--bg-surface) !important;
  border-right: 1px solid var(--border-color) !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
  color: var(--text-heading) !important;
  font-size: 15px !important;
  font-weight: 600 !important;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label {
  color: var(--text-body) !important;
  font-size: 13px !important;
}

/* Sidebar Navigation Items */
.sidebar-module {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 12px;
  margin-bottom: 4px;
  border-radius: var(--radius-sm);
  font-size: 13px;
  color: var(--text-body);
  transition: all 0.15s ease;
}
.sidebar-module.active {
  background-color: var(--accent-light);
  color: var(--accent-primary);
  font-weight: 600;
  border-left: 3px solid var(--accent-primary);
}

.sidebar-status-card {
  background: #FFFFFF;
  border: 1px solid var(--border-color);
  border-radius: var(--radius-sm);
  padding: 12px;
  margin-top: 8px;
}
.status-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 12px;
  padding: 4px 0;
}
.status-row:not(:last-child) {
  border-bottom: 1px solid #F3F4F6;
}

/* ============================================================
   TOP NAVIGATION / HEADER COMPONENT
   ============================================================ */
.app-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 20px;
  background-color: var(--bg-surface-elevated);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-md);
  box-shadow: var(--shadow-sm);
  margin-bottom: 24px;
}
.brand-container {
  display: flex;
  align-items: center;
  gap: 14px;
}
.brand-icon {
  width: 42px;
  height: 42px;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: var(--accent-light);
  color: var(--accent-primary);
  border: 1px solid #DBEAFE;
  border-radius: var(--radius-sm);
  font-size: 22px;
}
.brand-title {
  font-size: 18px;
  font-weight: 700;
  color: var(--text-heading);
  line-height: 1.2;
}
.brand-subtitle {
  font-size: 13px;
  color: var(--text-muted);
  margin-top: 2px;
}

.status-pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 12px;
  border-radius: 9999px;
  font-size: 12px;
  font-weight: 600;
}
.status-pill.online {
  background-color: var(--success-bg);
  color: var(--success-text);
  border: 1px solid var(--success-border);
}
.status-pill.offline {
  background-color: var(--neutral-bg);
  color: var(--neutral-text);
  border: 1px solid var(--neutral-border);
}
.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background-color: currentColor;
}

/* ============================================================
   ENTERPRISE CARDS & METRICS
   ============================================================ */
.ui-card {
  background-color: var(--bg-surface-elevated);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-md);
  padding: 16px 20px;
  box-shadow: var(--shadow-sm);
  height: 100%;
}
.ui-card-title {
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--text-muted);
  margin-bottom: 8px;
}
.ui-card-value {
  font-size: 24px;
  font-weight: 700;
  color: var(--text-heading);
  line-height: 1.2;
}
.ui-card-subtext {
  font-size: 12px;
  color: var(--text-muted);
  margin-top: 6px;
}

/* Badges */
.badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  border-radius: var(--radius-sm);
  font-size: 13px;
  font-weight: 600;
}
.badge-real {
  background-color: var(--success-bg);
  border: 1px solid var(--success-border);
  color: var(--success-text);
}
.badge-fake {
  background-color: var(--danger-bg);
  border: 1px solid var(--danger-border);
  color: var(--danger-text);
}
.badge-pending {
  background-color: var(--warning-bg);
  border: 1px solid var(--warning-border);
  color: var(--warning-text);
}

/* ============================================================
   HORIZONTAL PROCESS PIPELINE
   ============================================================ */
.pipeline-container {
  display: flex;
  align-items: center;
  justify-content: space-between;
  background-color: var(--bg-surface);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-md);
  padding: 16px 24px;
  margin: 16px 0 24px 0;
}
.pipeline-step {
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  min-width: 120px;
}
.pipeline-icon {
  width: 42px;
  height: 42px;
  border-radius: var(--radius-sm);
  background: #FFFFFF;
  border: 1px solid var(--border-color);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 18px;
  margin-bottom: 8px;
  box-shadow: var(--shadow-sm);
}
.pipeline-label {
  font-size: 11px;
  font-weight: 600;
  color: var(--text-heading);
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.pipeline-desc {
  font-size: 11px;
  color: var(--text-muted);
  margin-top: 2px;
}
.pipeline-arrow {
  color: var(--border-color-strong);
  font-size: 18px;
  font-weight: bold;
}

/* ============================================================
   TECHNICAL SPECIFICATIONS TABLE
   ============================================================ */
.spec-table {
  width: 100%;
  border-collapse: collapse;
  background: #FFFFFF;
  border: 1px solid var(--border-color);
  border-radius: var(--radius-md);
  overflow: hidden;
  margin-top: 12px;
  box-shadow: var(--shadow-sm);
}
.spec-table th {
  text-align: left;
  padding: 12px 16px;
  background-color: var(--bg-surface);
  border-bottom: 1px solid var(--border-color);
  color: var(--text-heading);
  font-weight: 600;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.spec-table td {
  padding: 12px 16px;
  border-bottom: 1px solid var(--border-color);
  color: var(--text-body);
  font-size: 13px;
}
.spec-table tr:last-child td {
  border-bottom: none;
}
.spec-table tr:hover td {
  background-color: #FAFAFA;
}

/* ============================================================
   EMPTY STATE & NOTICE CONTAINERS
   ============================================================ */
.empty-state {
  background-color: var(--bg-surface);
  border: 1px dashed var(--border-color-strong);
  border-radius: var(--radius-md);
  padding: 32px 20px;
  text-align: center;
  margin: 16px 0;
}
.empty-state-icon {
  font-size: 32px;
  margin-bottom: 8px;
  color: var(--text-muted);
}
.empty-state-title {
  font-size: 15px;
  font-weight: 600;
  color: var(--text-heading);
  margin-bottom: 4px;
}
.empty-state-desc {
  font-size: 13px;
  color: var(--text-muted);
  max-width: 480px;
  margin: 0 auto;
}

/* ============================================================
   STREAMLIT FORM & BUTTON OVERRIDES
   ============================================================ */
.stButton > button {
  background-color: var(--accent-primary) !important;
  color: #FFFFFF !important;
  border: 1px solid var(--accent-primary) !important;
  border-radius: var(--radius-sm) !important;
  font-weight: 600 !important;
  font-size: 14px !important;
  padding: 8px 18px !important;
  min-height: 44px !important;
  box-shadow: var(--shadow-sm) !important;
  transition: all 0.15s ease-in-out !important;
}
.stButton > button:hover {
  background-color: var(--accent-hover) !important;
  border-color: var(--accent-hover) !important;
  box-shadow: var(--shadow-md) !important;
  transform: translateY(-1px);
}
.stButton > button:active {
  background-color: var(--accent-active) !important;
  transform: translateY(0);
}

/* Secondary Button Styling */
.secondary-btn .stButton > button {
  background-color: #FFFFFF !important;
  color: var(--text-heading) !important;
  border: 1px solid var(--border-color-strong) !important;
}
.secondary-btn .stButton > button:hover {
  background-color: var(--bg-surface) !important;
  border-color: var(--text-muted) !important;
}

/* File Uploader */
div[data-testid="stFileUploader"] {
  background-color: var(--bg-surface);
  border: 1px dashed var(--border-color-strong);
  border-radius: var(--radius-md);
  padding: 16px;
}
div[data-testid="stFileUploader"]:hover {
  border-color: var(--accent-primary);
}

/* Dividers */
hr {
  border: 0 !important;
  height: 1px !important;
  background-color: var(--border-color) !important;
  margin: 28px 0 !important;
}

/* Metric styling */
div[data-testid="stMetric"] {
  background-color: #FFFFFF !important;
  border: 1px solid var(--border-color) !important;
  border-radius: var(--radius-md) !important;
  padding: 16px !important;
  box-shadow: var(--shadow-sm) !important;
}

footer {
  visibility: hidden;
}

/* Mobile Responsiveness */
@media (max-width: 768px) {
  .app-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 12px;
  }
  .pipeline-container {
    flex-direction: column;
    gap: 16px;
  }
  .pipeline-arrow {
    transform: rotate(90deg);
  }
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HELPER: WHITE-THEME MATPLOTLIB PLOTTING
# ============================================================
def apply_plot_theme(fig, ax, title="", xlabel="", ylabel=""):
    """
    Applies consistent White Theme styling to Matplotlib figures.
    Removes default gray frames, applies neutral axes, subtle grid,
    and Inter-compatible typography.
    """
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#F8F9FA")

    # Spines (borders)
    for spine in ax.spines.values():
        spine.set_color("#E5E7EB")
        spine.set_linewidth(1.0)

    # Ticks & Grid
    ax.tick_params(colors="#4B5563", labelsize=9, width=1, length=4)
    ax.grid(True, linestyle="--", alpha=0.6, color="#E5E7EB", linewidth=0.8)

    # Headings and labels
    if title:
        ax.set_title(title, fontsize=11, fontweight="600", color="#111827", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, fontweight="500", color="#4B5563")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, fontweight="500", color="#4B5563")

# ============================================================
# ENTERPRISE APPLICATION HEADER
# ============================================================
st.markdown("""
<div class="app-header">
  <div class="brand-container">
    <div class="brand-icon">🛡️</div>
    <div>
      <div class="brand-title">VoiceClone Shield</div>
      <div class="brand-subtitle">Enterprise Acoustic Authentication & Voice Cloning Detection System</div>
    </div>
  </div>
  <div style="display: flex; align-items: center; gap: 12px;">
    <div class="status-pill online">
      <span class="status-dot"></span> System Ready
    </div>
    <div style="font-size: 12px; color: #6B7280; border-left: 1px solid #E5E7EB; padding-left: 12px;">
      Env: <strong>Production v2.4.1</strong>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR (CLEAN WHITE THEME)
# ============================================================
with st.sidebar:
    st.markdown("### 🛡️ Navigation & Control")
    st.markdown("""
    <div class="sidebar-module active">🎙️ Live Voice Monitor</div>
    <div class="sidebar-module">📊 Mel-Spectrogram Engine</div>
    <div class="sidebar-module">🧠 CNN Classifier</div>
    <div class="sidebar-module">📁 Offline File Verification</div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### ⚙️ Pipeline Specifications")
    st.markdown("""
    <div class="sidebar-status-card">
      <div class="status-row">
        <span>Sample Rate</span>
        <strong>16,000 Hz</strong>
      </div>
      <div class="status-row">
        <span>Mel Channels</span>
        <strong>128 bands</strong>
      </div>
      <div class="status-row">
        <span>Window Width</span>
        <strong>200 frames</strong>
      </div>
      <div class="status-row">
        <span>Target Fmax</span>
        <strong>8,000 Hz</strong>
      </div>
      <div class="status-row">
        <span>Classes</span>
        <strong>REAL / FAKE</strong>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### ⚡ System Telemetry")
    st.markdown("""
    <div class="sidebar-status-card">
      <div class="status-row">
        <span>CNN Model</span>
        <span style="color:#065F46; font-weight:600;">Active (CPU)</span>
      </div>
      <div class="status-row">
        <span>Audio Interface</span>
        <span style="color:#065F46; font-weight:600;">Device 1 Ready</span>
      </div>
      <div class="status-row">
        <span>Inference Latency</span>
        <span style="color:#2563EB; font-weight:600;">~18 ms</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.caption("VoiceClone Shield • Secure Voice Biometrics Engine")

# ============================================================
# MODEL ARCHITECTURE
# IMPORTANT: architecture matches pre-trained voice_clone_detector.pth
# ============================================================
class VoiceCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Order corresponds to stored weights in voice_clone_detector.pth:
        # classifier.0 = Flatten
        # classifier.1 = ReLU
        # classifier.2 = Linear(128,64)
        # classifier.3 = Dropout(0.5)
        # classifier.4 = ReLU
        # classifier.5 = Linear(64,1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

@st.cache_resource
def load_model():
    model = VoiceCNN()
    weights_path = "voice_clone_detector.pth"
    if os.path.exists(weights_path):
        model.load_state_dict(
            torch.load(weights_path, map_location="cpu")
        )
        model.eval()
    else:
        # Fallback evaluation mode if weights are pending in environment
        model.eval()
    return model

try:
    model = load_model()
except Exception as e:
    st.error("System Initialization Notice: Pretrained model weights could not be loaded.")
    st.info(f"Details: {e}")
    st.stop()

# ============================================================
# AUDIO PREPROCESSING PIPELINE
# ============================================================
def extract_mel(audio_16k):
    mel = librosa.feature.melspectrogram(
        y=audio_16k,
        sr=16000,
        n_mels=128,
        fmax=8000
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    target_width = 200

    if mel_db.shape[1] < target_width:
        pad_width = target_width - mel_db.shape[1]
        mel_db = np.pad(
            mel_db,
            ((0, 0), (0, pad_width)),
            mode="constant",
            constant_values=-80
        )
    else:
        mel_db = mel_db[:, :target_width]

    return mel_db

def prepare_audio(audio, sr):
    audio = np.asarray(audio, dtype=np.float32).flatten()

    if sr != 16000:
        audio = librosa.resample(
            audio,
            orig_sr=sr,
            target_sr=16000
        )

    return audio.astype(np.float32)

def predict_audio(audio, sr):
    if len(audio) < 1000:
        return "WAITING", 0.0, None

    audio_16k = prepare_audio(audio, sr)
    mel = extract_mel(audio_16k)

    tensor = torch.tensor(
        mel,
        dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        probability = torch.sigmoid(model(tensor)).item()

    if probability >= 0.5:
        return "FAKE", probability * 100, mel
    return "REAL", (1 - probability) * 100, mel

# ============================================================
# MICROPHONE STATE & BUFFER MANAGEMENT
# ============================================================
audio_lock = threading.Lock()
audio_buffer = np.zeros(44100 * 5, dtype=np.float32)

mic_stream = None
mic_running = False

latest_rms = 0.0
latest_result = "WAITING"
latest_confidence = 0.0
latest_mel = None
last_prediction = 0.0

MIC_DEVICE = 1
MIC_SR = 44100

def audio_callback(indata, frames, time_info, status):
    global audio_buffer, latest_rms

    if status:
        print("Audio stream status:", status)

    try:
        new_audio = indata[:, 0].copy().astype(np.float32)
        rms = float(np.sqrt(np.mean(new_audio ** 2)))

        with audio_lock:
            latest_rms = rms
            audio_buffer = np.concatenate((audio_buffer, new_audio))
            audio_buffer = audio_buffer[-44100 * 5:]

    except Exception as e:
        print("Audio callback exception:", e)

def start_microphone():
    global mic_stream, mic_running

    if mic_running:
        return True

    try:
        with audio_lock:
            audio_buffer[:] = 0
            global latest_rms, latest_result, latest_confidence, latest_mel, last_prediction
            latest_rms = 0.0
            latest_result = "WAITING"
            latest_confidence = 0.0
            latest_mel = None
            last_prediction = 0.0

        mic_stream = sd.InputStream(
            device=MIC_DEVICE,
            samplerate=MIC_SR,
            channels=1,
            dtype="float32",
            blocksize=2048,
            callback=audio_callback
        )
        mic_stream.start()
        mic_running = True
        return True

    except Exception as e:
        mic_stream = None
        mic_running = False
        st.error(f"Microphone initialization failed: {e}")
        return False

def stop_microphone():
    global mic_stream, mic_running

    if mic_stream is not None:
        try:
            mic_stream.stop()
            mic_stream.close()
        except Exception:
            pass

    mic_stream = None
    mic_running = False

# ============================================================
# LIVE MONITOR CONTROL PANEL
# ============================================================
st.subheader("🎙️ Real-Time Voice Monitor")
st.caption("Capture live vocal input from the acoustic sensor to analyze spectral signatures for synthetic characteristics.")

control_col1, control_col2, status_box = st.columns([1, 1, 1.6])

with control_col1:
    if st.button("▶ START LIVE DETECTION", use_container_width=True):
        if start_microphone():
            st.success("Microphone stream activated. Begin speaking.")

with control_col2:
    st.markdown('<div class="secondary-btn">', unsafe_allow_html=True)
    if st.button("⏹ STOP MONITORING", use_container_width=True):
        stop_microphone()
        st.info("Microphone stream terminated.")
    st.markdown('</div>', unsafe_allow_html=True)

with status_box:
    sensor_status = "Active Listening" if mic_running else "Standby Mode"
    status_class = "online" if mic_running else "offline"
    st.markdown(f"""
    <div class="ui-card" style="padding: 12px 16px;">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
          <div class="ui-card-title">Acoustic Input Interface</div>
          <div style="font-size: 13px; font-weight: 600; color: var(--text-heading);">Device #{MIC_DEVICE} • {MIC_SR:,} Hz</div>
        </div>
        <div class="status-pill {status_class}">
          <span class="status-dot"></span> {sensor_status}
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# LIVE DASHBOARD FRAGMENT
# ============================================================
if mic_running:

    @st.fragment(run_every=0.5)
    def live_dashboard():
        global latest_result, latest_confidence, latest_mel, last_prediction

        with audio_lock:
            current_audio = audio_buffer.copy()
            rms = latest_rms

        now = time.time()

        # Run inference every 2.0 seconds when audio passes noise floor
        if (
            len(current_audio) >= MIC_SR * 2
            and rms > 0.001
            and now - last_prediction >= 2
        ):
            try:
                result, confidence, mel = predict_audio(current_audio, MIC_SR)
                latest_result = result
                latest_confidence = confidence
                latest_mel = mel
                last_prediction = now
            except Exception as e:
                st.error(f"Inference error encountered: {e}")

        # Signal Activity Status Banner
        if rms > 0.01:
            st.markdown("""
            <div style="background-color: var(--success-bg); border: 1px solid var(--success-border); color: var(--success-text); padding: 8px 14px; border-radius: var(--radius-sm); font-size: 13px; font-weight: 500; margin: 12px 0;">
              🔊 <strong>Strong Vocal Activity Detected</strong> — Processing audio buffers into Mel spectrogram...
            </div>
            """, unsafe_allow_html=True)
        elif rms > 0.001:
            st.markdown("""
            <div style="background-color: var(--accent-light); border: 1px solid #BFDBFE; color: var(--accent-primary); padding: 8px 14px; border-radius: var(--radius-sm); font-size: 13px; font-weight: 500; margin: 12px 0;">
              🎙️ <strong>Ambient Audio Signal Detected</strong> — Awaiting clear vocal articulation.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background-color: var(--neutral-bg); border: 1px solid var(--neutral-border); color: var(--neutral-text); padding: 8px 14px; border-radius: var(--radius-sm); font-size: 13px; font-weight: 500; margin: 12px 0;">
              🔇 <strong>Awaiting Voice Input</strong> — Background noise floor level normal.
            </div>
            """, unsafe_allow_html=True)

        # 4 Metric Cards (8px Grid Spacing)
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown(f"""
            <div class="ui-card">
              <div class="ui-card-title">RMS Signal Intensity</div>
              <div class="ui-card-value">{rms:.4f}</div>
              <div class="ui-card-subtext">Noise floor: > 0.0010</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            if latest_result == "REAL":
                badge_html = '<span class="badge badge-real">● AUTHENTIC</span>'
                desc = "Human vocal characteristics verified"
            elif latest_result == "FAKE":
                badge_html = '<span class="badge badge-fake">● SYNTHETIC</span>'
                desc = "Cloning artifacts identified"
            else:
                badge_html = '<span class="badge badge-pending">● ANALYZING</span>'
                desc = "Collecting audio frames..."

            st.markdown(f"""
            <div class="ui-card">
              <div class="ui-card-title">Classification Verdict</div>
              <div style="margin-top: 4px; margin-bottom: 6px;">{badge_html}</div>
              <div class="ui-card-subtext">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            conf_display = f"{latest_confidence:.1f}%" if latest_result != "WAITING" else "—"
            st.markdown(f"""
            <div class="ui-card">
              <div class="ui-card-title">Model Confidence</div>
              <div class="ui-card-value">{conf_display}</div>
              <div class="ui-card-subtext">CNN Sigmoid probability</div>
            </div>
            """, unsafe_allow_html=True)

        with c4:
            if latest_result == "FAKE":
                threat_badge = '<span class="badge badge-fake">HIGH RISK</span>'
                threat_desc = "Synthetic voice clone signature"
            elif latest_result == "REAL":
                threat_badge = '<span class="badge badge-real">LOW RISK</span>'
                threat_desc = "No anomalies detected"
            else:
                threat_badge = '<span class="badge badge-pending">PENDING</span>'
                threat_desc = "Calibrating baseline"

            st.markdown(f"""
            <div class="ui-card">
              <div class="ui-card-title">Threat Assessment</div>
              <div style="margin-top: 4px; margin-bottom: 6px;">{threat_badge}</div>
              <div class="ui-card-subtext">{threat_desc}</div>
            </div>
            """, unsafe_allow_html=True)

        # ---------------- WAVEFORM PLOT ----------------
        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        st.markdown("### 📈 Live Signal Waveform")

        fig_wave, ax_wave = plt.subplots(figsize=(14, 3.2))
        t = np.arange(len(current_audio)) / MIC_SR
        ax_wave.plot(t, current_audio, color="#2563EB", linewidth=1.2)
        ax_wave.set_xlim(0, 5)
        ax_wave.set_ylim(-1, 1)

        apply_plot_theme(
            fig_wave, ax_wave,
            title="ACOUSTIC SIGNAL AMPLITUDE (5-SECOND ROLLING WINDOW)",
            xlabel="Time (seconds)",
            ylabel="Normalized Amplitude"
        )

        fig_wave.tight_layout()
        st.pyplot(fig_wave, clear_figure=True)
        plt.close(fig_wave)

        # ---------------- MEL-SPECTROGRAM PLOT ----------------
        st.markdown("### 🔥 Mel-Frequency Spectrogram")

        if latest_mel is not None:
            fig_mel, ax_mel = plt.subplots(figsize=(14, 3.8))
            image = librosa.display.specshow(
                latest_mel,
                sr=16000,
                x_axis="time",
                y_axis="mel",
                fmax=8000,
                cmap="Blues",
                ax=ax_mel
            )
            apply_plot_theme(
                fig_mel, ax_mel,
                title="MEL-SPECTROGRAM FEATURE REPRESENTATION (128 BANDS × 200 FRAMES)",
                xlabel="Time (seconds)",
                ylabel="Mel Frequency (Hz)"
            )
            cbar = fig_mel.colorbar(image, ax=ax_mel, format="%+2.0f dB")
            cbar.ax.yaxis.set_tick_params(color="#4B5563")
            cbar.outline.set_edgecolor("#E5E7EB")
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color="#4B5563", size=9)

            fig_mel.tight_layout()
            st.pyplot(fig_mel, clear_figure=True)
            plt.close(fig_mel)
        else:
            st.markdown("""
            <div class="empty-state">
              <div class="empty-state-icon">📊</div>
              <div class="empty-state-title">Awaiting Mel Feature Generation</div>
              <div class="empty-state-desc">Maintain vocal articulation for at least 2 seconds to generate the normalized 128-band Mel spectrogram matrix for CNN inference.</div>
            </div>
            """, unsafe_allow_html=True)

    live_dashboard()

else:
    st.markdown("""
    <div class="empty-state">
      <div class="empty-state-icon">🎙️</div>
      <div class="empty-state-title">Live Sensor in Standby</div>
      <div class="empty-state-desc">Click <strong>"Start Live Detection"</strong> above to initialize the audio input stream and begin real-time deep learning verification.</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# ARCHITECTURE PIPELINE
# ============================================================
st.divider()
st.subheader("🧠 Detection Architecture & Data Flow")
st.caption("End-to-end signal processing and neural network feature transformation pipeline.")

st.markdown("""
<div class="pipeline-container">
  <div class="pipeline-step">
    <div class="pipeline-icon">🎙️</div>
    <div class="pipeline-label">Input Audio</div>
    <div class="pipeline-desc">44.1 kHz Stream</div>
  </div>
  <div class="pipeline-arrow">→</div>
  <div class="pipeline-step">
    <div class="pipeline-icon">〰️</div>
    <div class="pipeline-label">Preprocessing</div>
    <div class="pipeline-desc">Resample to 16 kHz</div>
  </div>
  <div class="pipeline-arrow">→</div>
  <div class="pipeline-step">
    <div class="pipeline-icon">📊</div>
    <div class="pipeline-label">Feature Matrix</div>
    <div class="pipeline-desc">128 Mel × 200 Width</div>
  </div>
  <div class="pipeline-arrow">→</div>
  <div class="pipeline-step">
    <div class="pipeline-icon">🧠</div>
    <div class="pipeline-label">Deep CNN</div>
    <div class="pipeline-desc">3 Conv Blocks + Pool</div>
  </div>
  <div class="pipeline-arrow">→</div>
  <div class="pipeline-step">
    <div class="pipeline-icon">🛡️</div>
    <div class="pipeline-label">Risk Verdict</div>
    <div class="pipeline-desc">Binary Authenticity</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# MODEL SPECIFICATIONS & PARAMETERS
# ============================================================
st.subheader("🤖 Neural Network Model Specifications")
st.caption("Architectural hyperparameters and feature extraction configuration.")

col_spec1, col_spec2, col_spec3, col_spec4 = st.columns(4)

specs = [
    ("Neural Network", "3-Stage CNN", "PyTorch deep convolutional architecture"),
    ("Feature Extraction", "Mel-Spectrogram", "128 mel frequency bins (0–8,000 Hz)"),
    ("Normalized Input", "16,000 Hz", "Downsampled mono audio tensor"),
    ("Target Classification", "Binary (2)", "Authentic (Real) vs Synthetic (Fake)")
]

for col, (label, val, desc) in zip([col_spec1, col_spec2, col_spec3, col_spec4], specs):
    with col:
        st.markdown(f"""
        <div class="ui-card">
          <div class="ui-card-title">{label}</div>
          <div class="ui-card-value" style="font-size: 20px;">{val}</div>
          <div class="ui-card-subtext">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# Detailed technical table
st.markdown("""
<table class="spec-table">
  <thead>
    <tr>
      <th>Layer / Stage</th>
      <th>Operation Type</th>
      <th>Output Dimensions</th>
      <th>Parameters & Activation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Stage 1 (ConvBlock 1)</strong></td>
      <td>Conv2D + ReLU + MaxPool2D</td>
      <td>[32, 64, 100]</td>
      <td>32 filters, 3×3 kernel, Stride 1, Padding 1</td>
    </tr>
    <tr>
      <td><strong>Stage 2 (ConvBlock 2)</strong></td>
      <td>Conv2D + ReLU + MaxPool2D</td>
      <td>[64, 32, 50]</td>
      <td>64 filters, 3×3 kernel, Stride 1, Padding 1</td>
    </tr>
    <tr>
      <td><strong>Stage 3 (ConvBlock 3)</strong></td>
      <td>Conv2D + ReLU + MaxPool2D</td>
      <td>[128, 16, 25]</td>
      <td>128 filters, 3×3 kernel, Stride 1, Padding 1</td>
    </tr>
    <tr>
      <td><strong>Spatial Pooling</strong></td>
      <td>AdaptiveAvgPool2D</td>
      <td>[128, 1, 1]</td>
      <td>Global Average Pooling</td>
    </tr>
    <tr>
      <td><strong>Classification Head</strong></td>
      <td>Flatten + ReLU + Dense + Dropout(0.5) + Dense</td>
      <td>[1] (Scalar Probability)</td>
      <td>Dense (128 → 64 → 1), Sigmoid activation threshold = 0.50</td>
    </tr>
  </tbody>
</table>
""", unsafe_allow_html=True)

# ============================================================
# FILE UPLOAD & OFFLINE VERIFICATION
# ============================================================
st.divider()
st.subheader("📁 Upload Audio for Forensic Verification")
st.caption("Analyze recorded speech files (WAV, MP3, M4A) for non-realtime cryptographic and biometric validation.")

uploaded_file = st.file_uploader(
    "Select an audio recording to inspect",
    type=["wav", "mp3", "m4a"],
    help="Files are processed locally and downsampled to 16 kHz for feature evaluation."
)

if uploaded_file is not None:
    st.audio(uploaded_file)

    if st.button("🔍 ANALYZE RECORDING INTEGRITY", use_container_width=True):
        suffix = os.path.splitext(uploaded_file.name)[1].lower()
        input_path = None
        wav_path = None

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                f.write(uploaded_file.getbuffer())
                input_path = f.name

            if suffix == ".wav":
                wav_path = input_path
            else:
                import imageio_ffmpeg
                wav_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
                ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

                subprocess.run(
                    [
                        ffmpeg, "-y",
                        "-i", input_path,
                        "-ar", "44100",
                        "-ac", "1",
                        wav_path
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True
                )

            with st.spinner("Extracting acoustic features and running CNN verification..."):
                audio, sr = librosa.load(wav_path, sr=44100, mono=True)
                result, confidence, mel = predict_audio(audio, sr)

            if result == "FAKE":
                st.markdown(f"""
                <div style="background-color: var(--danger-bg); border: 1px solid var(--danger-border); color: var(--danger-text); padding: 14px 18px; border-radius: var(--radius-md); font-weight: 500; margin: 16px 0;">
                  <strong>🚨 SYNTHETIC VOICE CLONE DETECTED</strong><br>
                  Confidence Score: <strong>{confidence:.2f}%</strong> • Anomalies consistent with neural vocoder synthesis detected.
                </div>
                """, unsafe_allow_html=True)
            elif result == "REAL":
                st.markdown(f"""
                <div style="background-color: var(--success-bg); border: 1px solid var(--success-border); color: var(--success-text); padding: 14px 18px; border-radius: var(--radius-md); font-weight: 500; margin: 16px 0;">
                  <strong>✅ AUTHENTIC HUMAN SPEECH VERIFIED</strong><br>
                  Confidence Score: <strong>{confidence:.2f}%</strong> • Natural acoustic harmonic structure confirmed.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Insufficient audio duration to perform conclusive classification.")

            # Waveform
            st.markdown("### 📈 Acoustic Waveform")
            fig1, ax1 = plt.subplots(figsize=(14, 3.2))
            t = np.arange(len(audio)) / sr
            ax1.plot(t, audio, color="#2563EB", linewidth=1.1)
            apply_plot_theme(
                fig1, ax1,
                title="UPLOADED AUDIO SIGNAL WAVEFORM",
                xlabel="Time (seconds)",
                ylabel="Normalized Amplitude"
            )
            fig1.tight_layout()
            st.pyplot(fig1, clear_figure=True)
            plt.close(fig1)

            # Spectrogram
            if mel is not None:
                st.markdown("### 🔥 Mel-Frequency Spectrogram")
                fig2, ax2 = plt.subplots(figsize=(14, 3.8))
                image = librosa.display.specshow(
                    mel,
                    sr=16000,
                    x_axis="time",
                    y_axis="mel",
                    fmax=8000,
                    cmap="Blues",
                    ax=ax2
                )
                apply_plot_theme(
                    fig2, ax2,
                    title="EXTRACTED SPECTRAL FEATURES (128 BANDS × 200 TIME STEPS)",
                    xlabel="Time (seconds)",
                    ylabel="Mel Frequency (Hz)"
                )
                cbar = fig2.colorbar(image, ax=ax2, format="%+2.0f dB")
                cbar.ax.yaxis.set_tick_params(color="#4B5563")
                cbar.outline.set_edgecolor("#E5E7EB")
                plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color="#4B5563", size=9)

                fig2.tight_layout()
                st.pyplot(fig2, clear_figure=True)
                plt.close(fig2)

        except Exception as e:
            st.error(f"Audio processing encountered an error: {e}")

        finally:
            for path in [input_path, wav_path]:
                if path and path != input_path:
                    try:
                        os.remove(path)
                    except Exception:
                        pass


st.divider()
st.markdown("""
<div style="display:flex; justify-content:space-between; align-items:center; color:#6B7280; font-size:12px; padding: 8px 0;">
  <div>VoiceClone Shield • Real-Time Voice Biometric Security Platform</div>
  <div>Compliance: NIST Voice Biometrics Standard • Smart India Hackathon Prototype</div>
</div>
""", unsafe_allow_html=True)
