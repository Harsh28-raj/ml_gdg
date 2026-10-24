"""Category classification: CLIP zero-shot (image) + keywords/MiniLM (text)."""
import logging
import re
from typing import Optional

import numpy as np

from app import config
from app.models_loader import (
    clip_image_embedding, clip_logit_scale, clip_text_embeddings, models, text_embeddings,
)
from app.schemas import ClassifyRequest, ClassifyResponse
from app.utils.image_io import load_image

log = logging.getLogger(__name__)
CATS = config.CATEGORIES

# Precomputed once after models load (see init()).
_clip_cat_emb: Optional[np.ndarray] = None   # [n_cats, 512], mean of prompt embeddings
_desc_emb: Optional[np.ndarray] = None       # [n_cats, 384]
_suffix = "(?:" + "|".join(config.KEYWORD_SUFFIXES) + ")?"
_kw_patterns: dict[str, list[re.Pattern]] = {
    c: [re.compile(r"\b" + re.escape(k.lower()) + _suffix + r"\b") for k in config.CATEGORY_KEYWORDS.get(c, [])]
    for c in CATS
}


def init() -> None:
    """Precompute CLIP prompt + category description embeddings. Safe to call repeatedly."""
    global _clip_cat_emb, _desc_emb
    if not models.loaded or _clip_cat_emb is not None:
        return
    rows = []
    for c in CATS:
        e = clip_text_embeddings(config.CLIP_PROMPTS[c]).mean(axis=0)  # prompt ensemble
        rows.append(e / np.linalg.norm(e))
    _clip_cat_emb = np.stack(rows)
    _desc_emb = text_embeddings([config.CATEGORY_DESCRIPTIONS[c] for c in CATS])
    log.info("Classify embeddings precomputed")


def _softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = x / temperature
    e = np.exp(z - z.max())
    return e / e.sum()


def image_scores(img) -> np.ndarray:
    """CLIP zero-shot probabilities over CATS."""
    sims = _clip_cat_emb @ clip_image_embedding(img)
    return _softmax(sims * clip_logit_scale())


def keyword_hits(text: str) -> np.ndarray:
    t = text.lower()
    return np.array([sum(1 for p in _kw_patterns[c] if p.search(t)) for c in CATS], dtype=np.float32)


def text_scores(text: str) -> np.ndarray:
    """Keyword match (normalized) + MiniLM similarity, softmax over CATS."""
    hits = keyword_hits(text)
    kw = hits / hits.max() if hits.max() > 0 else hits
    if models.loaded:
        sem = np.clip(_desc_emb @ text_embeddings([text])[0], 0.0, 1.0)
        raw = config.KEYWORD_WEIGHT * kw + config.SEMANTIC_WEIGHT * sem
    else:
        raw = kw  # keyword-only fallback
    return _softmax(raw, config.SOFTMAX_TEMPERATURE_TEXT)


def _label(conf: float) -> str:
    if conf > config.CONF_HIGH:
        return "High"
    return "Medium" if conf >= config.CONF_MEDIUM else "Low"


def classify(req: ClassifyRequest) -> ClassifyResponse:
    """Classify a complaint from text and/or image. Never raises."""
    init()
    reasons: list[str] = []
    text = (req.text or "").strip()
    img_given = bool(req.image_url or req.image_base64)

    img_p = txt_p = None
    if img_given:
        if not models.loaded:
            reasons.append("models not loaded; image ignored")
        else:
            img = load_image(req.image_url, req.image_base64)
            if img is None:
                reasons.append("image could not be loaded")
            else:
                try:
                    img_p = image_scores(img)
                except Exception as e:
                    log.exception("CLIP failed")
                    reasons.append(f"image scoring failed: {type(e).__name__}")
    if text:
        try:
            txt_p = text_scores(text)
        except Exception as e:
            log.exception("Text scoring failed")
            reasons.append(f"text scoring failed: {type(e).__name__}")

    if img_p is not None and txt_p is not None:
        probs, source = config.IMAGE_WEIGHT * img_p + config.TEXT_WEIGHT * txt_p, "text+image"
    elif img_p is not None:
        probs, source = img_p, "image"
    elif txt_p is not None:
        probs, source = txt_p, "text"
    else:
        reasons.append("no usable text or image")
        return ClassifyResponse(
            complaint_id=req.complaint_id, category="other", confidence=0.0, confidence_label="Low",
            scores={c: 0.0 for c in CATS}, source="none", needs_review=True, reason="; ".join(reasons),
        )

    i = int(np.argmax(probs))
    conf = float(probs[i])
    label = _label(conf)
    if label == "Low":
        reasons.append("low confidence")
    return ClassifyResponse(
        complaint_id=req.complaint_id,
        category=CATS[i],
        confidence=round(conf, 4),
        confidence_label=label,
        scores={c: round(float(p), 4) for c, p in zip(CATS, probs)},
        source=source,
        needs_review=label == "Low" or (img_given and img_p is None),
        reason="; ".join(reasons) or None,
    )
