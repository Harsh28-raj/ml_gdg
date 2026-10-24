"""Download + load CLIP and MiniLM once (used at build time so the server starts fast). Exits 1 on failure."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.models_loader import load_models, models  # noqa: E402

load_models()
if not models.loaded:
    sys.exit(f"Model loading failed: {models.error}")
print("Models downloaded and loaded OK")
