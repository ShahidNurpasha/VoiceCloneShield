# VoiceCloneShield

AI-powered real-time detection and prevention of voice-cloning impersonation attacks.

## Project files

- `app.py` — Streamlit application
- `advanced_models.py` — extended voice-cloning detection model architectures and training utilities
- `requirements.txt` — Python dependencies
- `.gitignore` — files that should not be committed
- `README.md` — project documentation

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Model weights

The application looks for `voice_clone_detector.pth` in the project root. If you use a trained model file, keep it local unless you intentionally want to store it in a suitable model-storage service.
