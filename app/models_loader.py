"""Loads CLIP + MiniLM ONCE at startup and exposes them as singletons."""
import logging
from typing import Optional

import numpy as np

from app import config

log = logging.getLogger(__name__)


class Models:
    clip_model = None
    clip_processor = None
    text_model = None
    loaded: bool = False
    error: Optional[str] = None


models = Models()


def load_models() -> None:
    """Load all models into the `models` singleton. Never raises; records error instead."""
    if models.loaded:
        return
    if not config.LOAD_MODELS:
        log.warning("ML_LOAD_MODELS=0 -> skipping model loading")
        return
    try:
        import torch
        from sentence_transformers import SentenceTransformer
        from transformers import CLIPModel, CLIPProcessor

        torch.set_grad_enabled(False)
        models.clip_model = CLIPModel.from_pretrained(config.CLIP_MODEL_NAME).to(config.DEVICE).eval()
        models.clip_processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL_NAME)
        models.text_model = SentenceTransformer(config.TEXT_MODEL_NAME, device=config.DEVICE)
        models.loaded = True
        from PIL import Image

        clip_image_embedding(Image.new("RGB", (64, 64)))  # warm-up so the first request is fast
        log.info("Models loaded: %s, %s", config.CLIP_MODEL_NAME, config.TEXT_MODEL_NAME)
    except Exception as e:  # keep server alive even if models fail
        models.error = f"{type(e).__name__}: {e}"
        log.exception("Model loading failed")


# ---------- Shared embedding helpers (models must be loaded) ----------
def _features(out):
    """transformers v5 returns an output object; older versions return a tensor."""
    return out if hasattr(out, "norm") else out.pooler_output


def _l2(x: "np.ndarray") -> "np.ndarray":
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


def clip_image_embeddings(imgs: list) -> "np.ndarray":
    """L2-normalized CLIP embeddings (shape [n, 512]) of PIL images, one batched forward pass."""
    import torch

    inputs = models.clip_processor(images=imgs, return_tensors="pt").to(config.DEVICE)
    with torch.no_grad():
        feats = _features(models.clip_model.get_image_features(**inputs))
    return _l2(feats.cpu().numpy().astype(np.float32))


def clip_image_embedding(img) -> "np.ndarray":
    """L2-normalized CLIP embedding (shape [512]) of one PIL image."""
    return clip_image_embeddings([img])[0]


def clip_text_embeddings(texts: list[str]) -> "np.ndarray":
    """L2-normalized CLIP text embeddings (shape [n, 512])."""
    import torch

    inputs = models.clip_processor(text=texts, return_tensors="pt", padding=True).to(config.DEVICE)
    with torch.no_grad():
        feats = _features(models.clip_model.get_text_features(**inputs))
    return _l2(feats.cpu().numpy().astype(np.float32))


def clip_logit_scale() -> float:
    return float(models.clip_model.logit_scale.exp().item())


def text_embeddings(texts: list[str]) -> "np.ndarray":
    """L2-normalized MiniLM sentence embeddings (shape [n, 384])."""
    return models.text_model.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
