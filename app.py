import os
import tempfile
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import streamlit as st
import torch
import torch.nn as nn

st.set_page_config(
    page_title="VoiceClone Shield",
    page_icon="🛡️",
    layout="wide"
)

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice_clone_detector.pth")
MODEL_OUTPUT_IS_FAKE = True
SAMPLE_RATE = 16000
N_MELS = 128
FMAX = 8000
TARGET_WIDTH = 200


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
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


@st.cache_resource
def load_model():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            "voice_clone_detector.pth was not found. Put it in the same folder as app.py."
        )

    model = VoiceCNN()

    try:
        state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(MODEL_PATH, map_location="cpu")

    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def extract_mel(audio):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        fmax=FMAX
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    if mel_db.shape[1] < TARGET_WIDTH:
        mel_db = np.pad(
            mel_db,
            ((0, 0), (0, TARGET_WIDTH - mel_db.shape[1])),
            mode="constant",
            constant_values=-80.0
        )
    else:
        mel_db = mel_db[:, :TARGET_WIDTH]

    return mel_db.astype(np.float32)


def prepare_audio(audio, sr):
    audio = np.asarray(audio, dtype=np.float32).flatten()

    if audio.size == 0:
        return audio

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak

    if sr != SAMPLE_RATE:
        audio = librosa.resample(
            audio,
            orig_sr=sr,
            target_sr=SAMPLE_RATE
        )

    return audio.astype(np.float32)


def predict_audio(audio, sr):
    audio_16k = prepare_audio(audio, sr)

    if len(audio_16k) < SAMPLE_RATE:
        return "WAITING", 0.0, None, None

    mel = extract_mel(audio_16k)
    tensor = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0)

    model = load_model()

    with torch.no_grad():
        logit = model(tensor).item()
        probability = float(torch.sigmoid(torch.tensor(logit)).item())

    if MODEL_OUTPUT_IS_FAKE:
        is_fake = probability >= 0.5
        fake_probability = probability
    else:
        is_fake = probability < 0.5
        fake_probability = 1.0 - probability

    if is_fake:
        return "FAKE", fake_probability * 100.0, mel, probability

    return "REAL", (1.0 - fake_probability) * 100.0, mel, probability


def plot_waveform(audio, sr):
    fig, ax = plt.subplots(figsize=(13, 3.5))
    time_axis = np.arange(len(audio)) / sr
    ax.plot(time_axis, audio, linewidth=1.0)
    ax.set_title("Recorded Voice Waveform")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Normalized Amplitude")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def plot_mel(mel):
    fig, ax = plt.subplots(figsize=(13, 4))
    image = librosa.display.specshow(
        mel,
        sr=SAMPLE_RATE,
        x_axis="time",
        y_axis="mel",
        fmax=FMAX,
        cmap="Blues",
        ax=ax
    )
    ax.set_title("Mel-Frequency Spectrogram")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Frequency")
    fig.colorbar(image, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    return fig


st.markdown(
    """
    <style>
    .block-container {max-width: 1250px; padding-top: 2rem;}
    .title {font-size: 2.2rem; font-weight: 800; margin-bottom: 0.2rem;}
    .subtitle {color: #64748b; margin-bottom: 1.5rem;}
    .card {padding: 18px; border: 1px solid #e2e8f0; border-radius: 12px; background: #fff;}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title">🛡️ VoiceClone Shield</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">AI-powered voice cloning and synthetic speech detection</div>',
    unsafe_allow_html=True
)

try:
    load_model()
    model_ready = True
except Exception as exc:
    model_ready = False
    st.error(f"Model loading failed: {exc}")

if model_ready:
    st.success("🟢 CNN model loaded successfully")
    st.caption("Model input: 16 kHz mono audio → 128-band Mel spectrogram → 200 frames → CNN")

st.divider()

tab_live, tab_file, tab_about = st.tabs(
    ["🎙️ Live Voice Monitor", "📁 Offline File Verification", "🧠 Model Information"]
)

with tab_live:
    st.subheader("Record your voice")
    st.write("Use your browser microphone, stop the recording, and the app will analyze it.")

    recording = st.audio_input("🎙️ Record your voice")

    if recording is not None and model_ready:
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_file.write(recording.getvalue())
                temp_path = temp_file.name

            with st.spinner("Analyzing audio..."):
                audio, sr = librosa.load(temp_path, sr=SAMPLE_RATE, mono=True)
                result, confidence, mel, raw_probability = predict_audio(audio, sr)

            if result == "FAKE":
                st.error(f"🚨 SYNTHETIC VOICE CLONE DETECTED — {confidence:.2f}% confidence")
            elif result == "REAL":
                st.success(f"✅ AUTHENTIC HUMAN SPEECH — {confidence:.2f}% confidence")
            else:
                st.warning("Audio is too short for analysis.")

            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(plot_waveform(audio, sr), clear_figure=True)
            with col2:
                if mel is not None:
                    st.pyplot(plot_mel(mel), clear_figure=True)

            if raw_probability is not None:
                st.caption(
                    f"Raw model sigmoid output: {raw_probability:.4f}. "
                    f"Current label mapping: 1 = {'FAKE' if MODEL_OUTPUT_IS_FAKE else 'REAL'}."
                )
        except Exception as exc:
            st.error(f"Microphone processing error: {exc}")
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

with tab_file:
    st.subheader("Upload an audio file")
    st.write("Supported formats: WAV, MP3 and M4A.")

    uploaded = st.file_uploader(
        "Choose an audio file",
        type=["wav", "mp3", "m4a"]
    )

    if uploaded is not None and model_ready:
        st.audio(uploaded)

        if st.button("🔍 Analyze Recording", use_container_width=True):
            input_path = None
            wav_path = None
            try:
                suffix = os.path.splitext(uploaded.name)[1].lower()

                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                    f.write(uploaded.getbuffer())
                    input_path = f.name

                if suffix == ".wav":
                    wav_path = input_path
                else:
                    import imageio_ffmpeg
                    wav_path = tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".wav"
                    ).name
                    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
                    import subprocess
                    subprocess.run(
                        [
                            ffmpeg, "-y",
                            "-i", input_path,
                            "-ar", str(SAMPLE_RATE),
                            "-ac", "1",
                            wav_path
                        ],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        check=True
                    )

                audio, sr = librosa.load(
                    wav_path,
                    sr=SAMPLE_RATE,
                    mono=True
                )

                result, confidence, mel, raw_probability = predict_audio(audio, sr)

                if result == "FAKE":
                    st.error(f"🚨 SYNTHETIC VOICE CLONE DETECTED — {confidence:.2f}% confidence")
                elif result == "REAL":
                    st.success(f"✅ AUTHENTIC HUMAN SPEECH — {confidence:.2f}% confidence")
                else:
                    st.warning("Audio is too short for analysis.")

                st.pyplot(plot_waveform(audio, sr), clear_figure=True)

                if mel is not None:
                    st.pyplot(plot_mel(mel), clear_figure=True)

                if raw_probability is not None:
                    st.caption(f"Raw model sigmoid output: {raw_probability:.4f}")
            except Exception as exc:
                st.error(f"File processing error: {exc}")
            finally:
                if input_path and os.path.exists(input_path) and input_path != wav_path:
                    os.remove(input_path)
                if wav_path and wav_path != input_path and os.path.exists(wav_path):
                    os.remove(wav_path)

with tab_about:
    st.subheader("Detection Pipeline")
    st.markdown(
        """
        1. **Audio input** — browser microphone or uploaded recording.
        2. **Preprocessing** — mono audio is resampled to 16 kHz.
        3. **Feature extraction** — a 128-band Mel spectrogram is created.
        4. **Fixed input** — the spectrogram is padded/cropped to 200 time frames.
        5. **CNN inference** — three convolutional blocks extract acoustic features.
        6. **Classification** — the final sigmoid produces a binary score.
        """
    )

    st.info(
        "Important: the supplied checkpoint stores model weights, but it does not store "
        "the original training label names. This project therefore uses the configured "
        "mapping 1 = FAKE. Validate this mapping with a known REAL and a known FAKE sample "
        "before presenting accuracy claims."
    )

    st.code(
        "voice_clone_detector.pth\n"
        "app.py\n"
        "advanced_models.py\n"
        "requirements.txt\n"
        "README.md\n"
        ".gitignore",
        language="text"
    )
