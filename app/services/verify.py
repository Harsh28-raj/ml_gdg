"""Before/after resolution verification: GPS, time, pHash copy check, CLIP same-place + issue-removed."""
import logging
from typing import Optional

import imagehash
import numpy as np

from app import config
from app.models_loader import clip_image_embeddings, clip_logit_scale, clip_text_embeddings, models
from app.schemas import VerifyChecks, VerifyRequest, VerifyResponse
from app.utils.geo import haversine_m
from app.utils.image_io import load_image

log = logging.getLogger(__name__)

# Precomputed once after models load: category -> [2, 512] (row 0 = issue present, row 1 = fixed).
_issue_emb: dict[str, np.ndarray] = {}


def init() -> None:
    """Precompute issue present/fixed prompt embeddings. Safe to call repeatedly."""
    if not models.loaded or _issue_emb:
        return
    for cat, p in config.VERIFY_PROMPTS.items():
        rows = [clip_text_embeddings(p[k]).mean(axis=0) for k in ("present", "fixed")]
        _issue_emb[cat] = np.stack([r / np.linalg.norm(r) for r in rows])
    log.info("Verify embeddings precomputed")


def _issue_prob(emb: np.ndarray, cat: str) -> float:
    """P(issue present) vs P(fixed) for one image embedding."""
    z = (_issue_emb[cat] @ emb) * clip_logit_scale()
    e = np.exp(z - z.max())
    return float(e[0] / e.sum())


def _time_ok(req: VerifyRequest) -> Optional[bool]:
    a, b = req.complaint_created_at, req.after_taken_at
    if a is None or b is None:
        return None
    try:
        return b > a
    except TypeError:  # naive vs aware datetimes
        return b.replace(tzinfo=None) > a.replace(tzinfo=None)


def verify(req: VerifyRequest) -> VerifyResponse:
    """Decide Verified / Rejected / Suspicious. Never raises."""
    init()
    checks = VerifyChecks()
    reasons: list[str] = []
    reject = False

    # 1. Location
    d = haversine_m(req.complaint_lat, req.complaint_lng, req.after_lat, req.after_lng)
    if d is None:
        reasons.append("missing GPS for complaint or after photo")
    else:
        checks.distance_m = round(d, 1)
        checks.location_ok = d <= config.VERIFY_MAX_DISTANCE_M
        if not checks.location_ok:
            reject = True
            reasons.append(f"after photo taken {d:.0f} m away (max {config.VERIFY_MAX_DISTANCE_M:.0f} m)")

    # 2. Time
    t_ok = _time_ok(req)
    if t_ok is None:
        reasons.append("missing complaint_created_at or after_taken_at")
    else:
        checks.time_ok = t_ok
        if not t_ok:
            reject = True
            reasons.append("after photo is not later than the complaint")

    # 3–5. Images
    before = load_image(req.before_image_url, req.before_image_base64)
    after = load_image(req.after_image_url, req.after_image_base64)
    if before is None:
        reasons.append("before image missing or unreadable")
    if after is None:
        reasons.append("after image missing or unreadable")

    if before is not None and after is not None:
        checks.phash_distance = int(imagehash.phash(before) - imagehash.phash(after))
        checks.not_same_photo = checks.phash_distance > config.PHASH_SAME_PHOTO_MAX
        if not checks.not_same_photo:
            reject = True
            reasons.append(f"before and after are the same photo (pHash distance {checks.phash_distance})")

        if not models.loaded:
            reasons.append("models not loaded; image checks skipped")
        else:
            cat = req.category if req.category in _issue_emb else "other"
            if cat != req.category:
                reasons.append(f"unknown category {req.category!r}; generic issue prompts used")
            try:
                eb, ea = clip_image_embeddings([before, after])
                checks.same_place_score = round(float(np.dot(eb, ea)), 4)
                checks.issue_before = round(_issue_prob(eb, cat), 4)
                checks.issue_after = round(_issue_prob(ea, cat), 4)
                checks.issue_removed = (
                    checks.issue_after < config.ISSUE_AFTER_MAX
                    and checks.issue_before - checks.issue_after >= config.ISSUE_DROP_MIN
                )
                if checks.same_place_score < config.SAME_PLACE_MIN:
                    reasons.append(f"after photo does not look like the same place ({checks.same_place_score:.2f})")
                if not checks.issue_removed:
                    reasons.append("issue does not look removed in after photo")
            except Exception as e:
                log.exception("CLIP verify failed")
                reasons.append(f"image scoring failed: {type(e).__name__}")

    same_place = checks.same_place_score is not None and checks.same_place_score >= config.SAME_PLACE_MIN
    w = config.VERIFY_WEIGHTS
    score = (w["location"] * checks.location_ok + w["time"] * checks.time_ok
             + w["not_same_photo"] * bool(checks.not_same_photo) + w["same_place"] * same_place
             + w["issue_removed"] * bool(checks.issue_removed))

    if reject:
        status = "Rejected"
    elif same_place and checks.issue_removed and checks.location_ok and checks.time_ok:
        status = "Verified"
    else:
        status = "Suspicious"

    return VerifyResponse(
        complaint_id=req.complaint_id,
        status=status,
        verification_score=round(score, 4),
        checks=checks,
        reasons=reasons,
        needs_review=status == "Suspicious",
        reason=None if status == "Verified" else (reasons[0] if reasons else status),
    )
