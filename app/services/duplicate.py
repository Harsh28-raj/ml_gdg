"""Duplicate detection: geo/time/category filter + MiniLM text sim + CLIP image sim."""
import logging
from typing import Optional

import numpy as np

from app import config
from app.models_loader import clip_image_embeddings, models, text_embeddings
from app.schemas import ComplaintItem, DuplicateDetails, DuplicateRequest, DuplicateResponse, TopMatch
from app.utils import cache
from app.utils.geo import haversine_m
from app.utils.image_io import load_image

log = logging.getLogger(__name__)


def _text_embs(items: list[ComplaintItem]) -> dict[str, np.ndarray]:
    """Text embeddings by complaint_id (cached; uncached ones encoded in one batch)."""
    out, todo = {}, []
    for it in items:
        if not (it.text or "").strip():
            continue
        v = cache.get_embedding(it.complaint_id, "text")
        if v is not None:
            out[it.complaint_id] = v
        else:
            todo.append(it)
    if todo:
        for it, v in zip(todo, text_embeddings([it.text.strip() for it in todo])):
            cache.set_embedding(it.complaint_id, "text", v)
            out[it.complaint_id] = v
    return out


def _image_embs(items: list[ComplaintItem]) -> dict[str, np.ndarray]:
    """CLIP image embeddings by complaint_id (cached; uncached ones in one batch). Unreadable images skipped."""
    out, todo, imgs = {}, [], []
    for it in items:
        if not (it.image_url or it.image_base64):
            continue
        v = cache.get_embedding(it.complaint_id, "image")
        if v is not None:
            out[it.complaint_id] = v
            continue
        img = load_image(it.image_url, it.image_base64)
        if img is not None:
            todo.append(it)
            imgs.append(img)
    if imgs:
        for it, v in zip(todo, clip_image_embeddings(imgs)):
            cache.set_embedding(it.complaint_id, "image", v)
            out[it.complaint_id] = v
    return out


def _days_apart(a: ComplaintItem, b: ComplaintItem) -> Optional[float]:
    if a.created_at is None or b.created_at is None:
        return None
    try:
        return abs((a.created_at - b.created_at).total_seconds()) / 86400
    except TypeError:  # naive vs aware datetimes
        return abs((a.created_at.replace(tzinfo=None) - b.created_at.replace(tzinfo=None)).total_seconds()) / 86400


def _filter(c: ComplaintItem, cands: list[ComplaintItem], reasons: list[str]) -> list[tuple[ComplaintItem, Optional[float], Optional[float]]]:
    """Keep candidates with same category, within distance and time window. Unknown values don't exclude."""
    kept, unknown = [], set()
    for k in cands:
        if k.complaint_id == c.complaint_id:
            continue
        if c.category and k.category and c.category != k.category:
            continue
        d = haversine_m(c.lat, c.lng, k.lat, k.lng)
        if d is None:
            unknown.add("GPS")
        elif d > config.DUP_MAX_DISTANCE_M:
            continue
        days = _days_apart(c, k)
        if days is None:
            unknown.add("created_at")
        elif days > config.DUP_MAX_DAYS:
            continue
        if not (c.category and k.category):
            unknown.add("category")
        kept.append((k, d, days))
    if kept and unknown:
        reasons.append("missing " + ", ".join(sorted(unknown)) + "; filter not fully applied")
    return kept


def find_duplicate(req: DuplicateRequest) -> DuplicateResponse:
    """Find the best matching earlier complaint. Never raises."""
    c = req.complaint
    cid = c.complaint_id
    if not req.candidates:
        return DuplicateResponse(complaint_id=cid, is_duplicate=False, reason="no candidates")
    if not models.loaded:
        return DuplicateResponse(complaint_id=cid, is_duplicate=False, needs_review=True, reason="models not loaded")

    reasons: list[str] = []
    kept = _filter(c, req.candidates, reasons)
    if not kept:
        return DuplicateResponse(complaint_id=cid, is_duplicate=False, reason="no candidates within category/distance/time window")

    try:
        texts = _text_embs([c] + [k for k, _, _ in kept])
        images = _image_embs([c] + [k for k, _, _ in kept]) if (c.image_url or c.image_base64) else {}
    except Exception as e:
        log.exception("Embedding failed")
        return DuplicateResponse(complaint_id=cid, is_duplicate=False, needs_review=True,
                                 reason=f"embedding failed: {type(e).__name__}")

    scored = []  # (score, candidate, details)
    for k, d, days in kept:
        t_sim = i_sim = None
        if cid in texts and k.complaint_id in texts:
            t_sim = float(np.dot(texts[cid], texts[k.complaint_id]))
        if cid in images and k.complaint_id in images:
            i_sim = float(np.dot(images[cid], images[k.complaint_id]))
        parts = [(config.DUP_TEXT_WEIGHT, t_sim), (config.DUP_IMAGE_WEIGHT, i_sim)]
        parts = [(w, s) for w, s in parts if s is not None]
        if not parts:
            continue
        score = sum(w * s for w, s in parts) / sum(w for w, _ in parts)
        if d is not None:
            score += config.DUP_DISTANCE_BONUS_MAX * max(0.0, 1 - d / config.DUP_MAX_DISTANCE_M)
        score = float(np.clip(score, 0.0, 1.0))
        details = DuplicateDetails(
            text_sim=None if t_sim is None else round(t_sim, 4),
            image_sim=None if i_sim is None else round(i_sim, 4),
            distance_m=None if d is None else round(d, 1),
            days_apart=None if days is None else round(days, 2),
        )
        scored.append((score, k, details))

    if not scored:
        return DuplicateResponse(complaint_id=cid, is_duplicate=False, needs_review=True,
                                 reason="no comparable text or image between complaint and candidates")

    scored.sort(key=lambda x: x[0], reverse=True)
    best, k, details = scored[0]
    top = [TopMatch(complaint_id=kk.complaint_id, score=round(s, 4), distance_m=dd.distance_m)
           for s, kk, dd in scored[:config.DUP_TOP_K]]
    is_dup = best >= config.DUP_THRESHOLD
    borderline = not is_dup and best >= config.DUP_BORDERLINE
    if borderline:
        reasons.append(f"borderline similarity {best:.2f}")
    return DuplicateResponse(
        complaint_id=cid,
        is_duplicate=is_dup,
        matched_complaint_id=k.complaint_id if (is_dup or borderline) else None,
        group_id=k.group_id if is_dup else None,
        similarity_score=round(best, 4),
        details=details,
        top_matches=top,
        needs_review=borderline or bool(reasons),
        reason="; ".join(reasons) or None,
    )
