"""Tests for /ml/duplicate."""
import base64
import io
import uuid
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from app.models_loader import models
from app.utils import cache

IMAGES = Path(__file__).resolve().parents[1] / "sample_data" / "images"
BASE = {"lat": 28.67, "lng": 77.45, "created_at": "2026-10-24T10:00:00", "category": "road"}
NEAR = {"lat": 28.6702, "lng": 77.4501, "created_at": "2026-10-22T09:00:00", "category": "road"}  # ~24 m, 2 days


@pytest.fixture(autouse=True)
def _need_models(client):
    if not models.loaded:
        pytest.skip("models not loaded (ML_LOAD_MODELS=0)")


def _id() -> str:
    return "T" + uuid.uuid4().hex[:8]


def _post(client, complaint, candidates):
    r = client.post("/ml/duplicate", json={"complaint": complaint, "candidates": candidates})
    assert r.status_code == 200
    return r.json()


def _b64_scene() -> str:
    im = Image.new("RGB", (320, 240), (90, 90, 95))
    ImageDraw.Draw(im).ellipse([100, 80, 220, 160], fill=(40, 30, 20))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _pair(text_a: str, text_b: str, **cand_over):
    a = {"complaint_id": _id(), "text": text_a, **BASE}
    b = {"complaint_id": _id(), "group_id": "G12", "text": text_b, **NEAR, **cand_over}
    return a, b


def test_same_issue_nearby_is_duplicate(client):
    a, b = _pair("Big pothole on the main road near the market", "Large pothole near market road, very dangerous")
    d = _post(client, a, [b])
    assert d["is_duplicate"] is True
    assert d["matched_complaint_id"] == b["complaint_id"]
    assert d["group_id"] == "G12"
    assert d["similarity_score"] >= 0.75
    assert 15 < d["details"]["distance_m"] < 35
    assert d["details"]["days_apart"] == pytest.approx(2.04, abs=0.01)
    assert d["needs_review"] is False


def test_far_away_not_duplicate(client):
    a, b = _pair("Big pothole near the market", "Big pothole near the market", lat=28.675)  # ~550 m
    d = _post(client, a, [b])
    assert d["is_duplicate"] is False and d["group_id"] is None


def test_different_category_not_duplicate(client):
    a, b = _pair("Big pothole near the market", "Big pothole near the market", category="waste")
    d = _post(client, a, [b])
    assert d["is_duplicate"] is False


def test_too_old_not_duplicate(client):
    a, b = _pair("Big pothole near the market", "Big pothole near the market", created_at="2026-10-01T09:00:00")
    assert _post(client, a, [b])["is_duplicate"] is False


def test_no_candidates(client):
    d = _post(client, {"complaint_id": _id(), "text": "pothole", **BASE}, [])
    assert d["is_duplicate"] is False and d["group_id"] is None


def test_unrelated_nearby_not_duplicate(client):
    a, b = _pair("Big pothole on the main road near the market", "Road is cracked near the school gate")
    d = _post(client, a, [b])
    assert d["is_duplicate"] is False and d["needs_review"] is False


def test_borderline_needs_review(client):
    a, b = _pair("sadak pe bada gaddha hai", "market ke paas sadak mein gaddha")
    d = _post(client, a, [b])
    assert d["is_duplicate"] is False
    assert 0.6 <= d["similarity_score"] < 0.75
    assert d["needs_review"] is True and d["matched_complaint_id"] == b["complaint_id"]


def test_best_of_several_and_top_matches(client):
    a, good = _pair("Big pothole on the main road near the market", "Large pothole near market road, very dangerous")
    _, bad = _pair("x", "Road is cracked near the school gate")
    d = _post(client, a, [bad, good])
    assert d["matched_complaint_id"] == good["complaint_id"]
    assert [m["complaint_id"] for m in d["top_matches"]] == [good["complaint_id"], bad["complaint_id"]]


def test_embeddings_cached(client, monkeypatch):
    a, b = _pair("Big pothole on the main road near the market", "Large pothole near market road, very dangerous")
    first = _post(client, a, [b])
    assert cache.get_embedding(b["complaint_id"], "text") is not None

    def boom(*_a, **_k):
        raise AssertionError("should use cache")
    monkeypatch.setattr("app.services.duplicate.text_embeddings", boom)
    assert _post(client, a, [b])["similarity_score"] == first["similarity_score"]


def test_image_similarity_used(client):
    img = _b64_scene()
    a = {"complaint_id": _id(), "image_base64": img, **BASE}
    b = {"complaint_id": _id(), "group_id": "G7", "image_base64": img, **NEAR}
    d = _post(client, a, [b])
    assert d["details"]["image_sim"] == pytest.approx(1.0, abs=1e-3)
    assert d["details"]["text_sim"] is None
    assert d["is_duplicate"] is True and d["group_id"] == "G7"
    assert cache.get_embedding(b["complaint_id"], "image") is not None


def test_real_pothole_photos_duplicate(client):
    for n in ("pothole1.jpg", "pothole1_b.jpg"):
        if not (IMAGES / n).exists():
            pytest.skip(f"missing sample image: sample_data/images/{n}")
    a = {"complaint_id": _id(), "image_url": "sample_data/images/pothole1.jpg", **BASE}
    b = {"complaint_id": _id(), "group_id": "G1", "image_url": "sample_data/images/pothole1_b.jpg", **NEAR}
    assert _post(client, a, [b])["is_duplicate"] is True


def _img_pair(a_name: str, b_name: str, category: str):
    for n in (a_name, b_name):
        if not (IMAGES / n).exists():
            pytest.skip(f"missing sample image: sample_data/images/{n}")
    a = {"complaint_id": _id(), "image_url": f"sample_data/images/{a_name}", **{**BASE, "category": category}}
    b = {"complaint_id": _id(), "group_id": "G1", "image_url": f"sample_data/images/{b_name}", **{**NEAR, "category": category}}
    return a, b


def test_real_garbage_same_spot_other_angle_duplicate(client):
    a, b = _img_pair("garbage2_a.jpg", "garbage2_b.jpg", "waste")
    d = _post(client, a, [b])
    assert d["is_duplicate"] is True and d["details"]["image_sim"] > 0.8


def test_real_different_garbage_spots_not_duplicate(client):
    a, b = _img_pair("garbage1.jpg", "garbage3.jpg", "waste")
    d = _post(client, a, [b])
    assert d["is_duplicate"] is False and d["details"]["image_sim"] < 0.7


def test_missing_gps_flags_review(client):
    a, b = _pair("Big pothole near the market", "Big pothole near the market")
    a.pop("lat"), a.pop("lng")
    d = _post(client, a, [b])
    assert d["needs_review"] is True and "GPS" in d["reason"]


def test_bad_image_and_no_text_no_crash(client):
    a = {"complaint_id": _id(), "image_url": "http://invalid.invalid/a.jpg", **BASE}
    b = {"complaint_id": _id(), "image_url": "nope.jpg", **NEAR}
    d = _post(client, a, [b])
    assert d["is_duplicate"] is False and d["needs_review"] is True


def test_self_in_candidates_ignored(client):
    a = {"complaint_id": _id(), "text": "pothole", **BASE}
    assert _post(client, a, [a])["is_duplicate"] is False
