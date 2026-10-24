"""Tests for /ml/verify."""
import base64
import io
from pathlib import Path

import pytest
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from app.models_loader import models

IMAGES = Path(__file__).resolve().parents[1] / "sample_data" / "images"
OK_META = {
    "complaint_id": "C1023", "category": "road",
    "complaint_lat": 28.67, "complaint_lng": 77.45, "after_lat": 28.6701, "after_lng": 77.4502,  # ~22 m
    "complaint_created_at": "2026-10-22T09:00:00", "after_taken_at": "2026-10-24T12:00:00",
}


@pytest.fixture(autouse=True)
def _need_models(client):
    if not models.loaded:
        pytest.skip("models not loaded (ML_LOAD_MODELS=0)")


def _post(client, **body):
    r = client.post("/ml/verify", json={**OK_META, **body})
    assert r.status_code == 200
    return r.json()


def _b64(im: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    im.save(buf, fmt)
    return base64.b64encode(buf.getvalue()).decode()


def _scene(color=(40, 30, 20)) -> Image.Image:
    im = Image.new("RGB", (320, 240), (90, 90, 95))
    d = ImageDraw.Draw(im)
    d.ellipse([100, 80, 220, 160], fill=color)
    d.rectangle([0, 200, 320, 240], fill=(60, 120, 60))
    return im


def _image(name: str) -> str:
    if not (IMAGES / name).exists():
        pytest.skip(f"missing sample image: sample_data/images/{name}")
    return f"sample_data/images/{name}"


def _different_pair() -> dict:
    """Two visibly different images so the pHash copy check passes."""
    b = _scene()
    a = Image.new("RGB", (320, 240), (200, 200, 205))
    ImageDraw.Draw(a).line([0, 0, 320, 240], fill=(0, 0, 0), width=20)
    return {"before_image_base64": _b64(b), "after_image_base64": _b64(a)}


@pytest.mark.parametrize("before,after,cat", [
    ("pothole1.jpg", "pothole1_fixed.jpg", "road"),
    ("garbage1.jpg", "garbage1_clean.jpg", "waste"),
])
def test_real_before_after_verified(client, before, after, cat):
    d = _post(client, category=cat, before_image_url=_image(before), after_image_url=_image(after))
    assert d["status"] == "Verified", d
    assert d["needs_review"] is False


def test_unrelated_clean_photo_suspicious(client):
    d = _post(client, category="waste", before_image_url=_image("garbage1.jpg"),
              after_image_url=_image("clean_road_other.jpg"))
    assert d["status"] == "Suspicious" and d["needs_review"] is True


def test_real_pothole_to_clean_road_elsewhere_suspicious(client):
    d = _post(client, category="road", before_image_url=_image("pothole1.jpg"),
              after_image_url=_image("clean_road_other.jpg"))
    assert d["status"] == "Suspicious"
    assert d["checks"]["issue_removed"] is True and d["checks"]["same_place_score"] < 0.75  # fixed-looking but elsewhere


def test_real_same_spot_issue_still_there_suspicious(client):
    d = _post(client, category="road", before_image_url=_image("pothole1.jpg"),
              after_image_url=_image("pothole1_b.jpg"))  # re-shot, pothole unchanged
    assert d["status"] == "Suspicious"
    assert d["checks"]["not_same_photo"] is True and d["checks"]["issue_removed"] is False


def test_real_same_photo_rejected(client):
    d = _post(client, before_image_url=_image("pothole1.jpg"), after_image_url=_image("pothole1.jpg"))
    assert d["status"] == "Rejected"


def test_same_photo_rejected(client):
    img = _b64(_scene())
    d = _post(client, before_image_base64=img, after_image_base64=img)
    assert d["status"] == "Rejected"
    assert d["checks"]["not_same_photo"] is False and d["checks"]["phash_distance"] == 0


def test_recompressed_copy_rejected(client):
    # textured image: pHash is unstable on flat synthetic images, real photos behave like this
    noise = (np.random.default_rng(0).random((60, 80, 3)) * 255).astype("uint8")
    im = Image.fromarray(noise).resize((320, 240), Image.BICUBIC).filter(ImageFilter.GaussianBlur(2))
    d = _post(client, before_image_base64=_b64(im), after_image_base64=_b64(im.resize((640, 480)), "JPEG"))
    assert d["status"] == "Rejected"
    assert d["checks"]["phash_distance"] <= 5


def test_far_gps_rejected(client):
    d = _post(client, after_lat=28.68, after_lng=77.46, **_different_pair())  # ~1.4 km
    assert d["status"] == "Rejected"
    assert d["checks"]["location_ok"] is False and d["checks"]["distance_m"] > 1000


def test_after_before_complaint_rejected(client):
    d = _post(client, after_taken_at="2026-10-20T09:00:00", **_different_pair())
    assert d["status"] == "Rejected" and d["checks"]["time_ok"] is False


def test_missing_gps_suspicious(client):
    d = _post(client, after_lat=None, after_lng=None, **_different_pair())
    assert d["status"] == "Suspicious" and d["needs_review"] is True
    assert d["checks"]["location_ok"] is False
    assert any("GPS" in r for r in d["reasons"])


def test_image_checks_populated(client):
    d = _post(client, **_different_pair())
    c = d["checks"]
    assert c["not_same_photo"] is True and c["phash_distance"] > 5
    assert 0 <= c["issue_before"] <= 1 and 0 <= c["issue_after"] <= 1
    assert -1 <= c["same_place_score"] <= 1
    assert 0 <= d["verification_score"] <= 1


@pytest.mark.parametrize("body", [
    {},
    {"before_image_url": "http://invalid.invalid/b.jpg", "after_image_url": "nope.jpg"},
    {"before_image_base64": "!!bad!!", "after_image_base64": "!!bad!!"},
])
def test_missing_or_bad_images_no_crash(client, body):
    d = _post(client, **body)
    assert d["status"] == "Suspicious" and d["needs_review"] is True
    assert d["checks"]["phash_distance"] is None


def test_unknown_category_no_crash(client):
    d = _post(client, category="noise", **_different_pair())
    assert d["status"] in ("Suspicious", "Verified")
    assert any("unknown category" in r for r in d["reasons"])
