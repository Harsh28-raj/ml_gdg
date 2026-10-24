"""Priority scoring: rule-based and explainable (category base + urgent keywords + report count)."""
import re

from app import config
from app.schemas import PriorityRequest, PriorityResponse

_suffix = "(?:" + "|".join(config.URGENT_KEYWORD_SUFFIXES) + ")?"
_urgent = [(k, re.compile(r"\b" + re.escape(k.lower()) + _suffix + r"\b")) for k in config.URGENT_KEYWORDS]


def _label(score: float) -> str:
    if score >= config.PRIORITY_HIGH:
        return "High"
    return "Medium" if score >= config.PRIORITY_MEDIUM else "Low"


def score_priority(req: PriorityRequest) -> PriorityResponse:
    """Score 0–1 → High/Medium/Low with a reason per boost. Never raises."""
    reasons: list[str] = []
    review: list[str] = []

    cat = req.category if req.category in config.CATEGORY_BASE_WEIGHT else "other"
    if cat != req.category:
        review.append(f"unknown category {req.category!r}; base weight of 'other' used")
    score = config.CATEGORY_BASE_WEIGHT[cat]

    text = (req.text or "").strip().lower()
    if not text:
        review.append("no text; urgent keywords not checked")
    hits = [k for k, p in _urgent if p.search(text)]
    if hits:
        score += min(config.URGENT_BOOST_CAP, config.URGENT_BOOST + config.URGENT_EXTRA_PER * (len(hits) - 1))
        reasons += [f"urgent keyword: {k}" for k in hits]

    n = max(0, req.duplicate_count)
    if n > 0:
        score += min(config.DUP_COUNT_BOOST_CAP, config.DUP_COUNT_BOOST_PER * n)
        reasons.append(f"{n} {'person' if n == 1 else 'people'} reported this")

    score = round(min(1.0, score), 4)
    return PriorityResponse(
        complaint_id=req.complaint_id,
        priority=_label(score),
        priority_score=score,
        reasons=reasons,
        needs_review=bool(review),
        reason="; ".join(review) or None,
    )
