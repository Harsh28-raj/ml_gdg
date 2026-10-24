"""Tests for /ml/classify."""
import base64
import io
import time
from pathlib import Path

import pytest
from PIL import Image

from app.models_loader import models

IMAGES = Path(__file__).resolve().parents[1] / "sample_data" / "images"


@pytest.fixture(autouse=True)
def _need_models(client):
    if not models.loaded:
        pytest.skip("models not loaded (ML_LOAD_MODELS=0)")


def _post(client, **body):
    r = client.post("/ml/classify", json={"complaint_id": "T1", **body})
    assert r.status_code == 200
    return r.json()


def _image(name: str) -> str:
    path = IMAGES / name
    if not path.exists():
        pytest.skip(f"missing sample image: sample_data/images/{name}")
    return f"sample_data/images/{name}"


def _b64_gray() -> str:
    buf = io.BytesIO()
    Image.new("RGB", (224, 224), (120, 120, 120)).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.mark.parametrize("name,expected", [
    ("garbage1.jpg", "waste"), ("garbage3.jpg", "waste"), ("pothole1.jpg", "road"),
    ("water1.jpg", "water"), ("water2.jpg", "water"), ("clean_road_other.jpg", "road"),
])
def test_image_only(client, name, expected):
    d = _post(client, image_url=_image(name))
    assert d["category"] == expected
    assert d["source"] == "image"


def test_image_and_text(client):
    d = _post(client, text="Gali mein kachra pada hai", image_url=_image("garbage1.jpg"))
    assert d["category"] == "waste"
    assert d["source"] == "text+image"
    assert d["needs_review"] is False


@pytest.mark.parametrize("text,expected", [
    ("Gali mein kachra pada hai", "waste"),
    ("sadak pe gaddha", "road"),
    ("street light not working", "streetlight"),
    ("pipe se paani leak ho raha hai", "water"),
])
def test_text_only(client, text, expected):
    d = _post(client, text=text)
    assert d["category"] == expected
    assert d["source"] == "text"
    assert d["confidence_label"] == "High"
    assert d["needs_review"] is False
    assert abs(sum(d["scores"].values()) - 1) < 1e-3


def test_image_only_base64_no_crash(client):
    d = _post(client, image_base64=_b64_gray())
    assert d["source"] == "image"
    assert set(d["scores"]) == {"waste", "road", "streetlight", "water", "other"}


def test_both_empty(client):
    d = _post(client, text="  ")
    assert d["category"] == "other"
    assert d["confidence"] == 0
    assert d["source"] == "none"
    assert d["needs_review"] is True


@pytest.mark.parametrize("body", [
    {"image_url": "sample_data/images/does_not_exist.jpg"},
    {"image_url": "http://invalid.invalid/x.jpg"},
    {"image_base64": "!!not-base64!!"},
])
def test_bad_image_no_crash(client, body):
    d = _post(client, **body)
    assert d["source"] == "none"
    assert d["needs_review"] is True
    assert "image could not be loaded" in d["reason"]


def test_bad_image_falls_back_to_text(client):
    d = _post(client, text="kachra", image_url="sample_data/images/does_not_exist.jpg")
    assert d["category"] == "waste"
    assert d["source"] == "text"
    assert d["needs_review"] is True
    assert "image could not be loaded" in d["reason"]


def test_low_confidence_needs_review(client):
    d = _post(client, text="hello")
    assert d["confidence_label"] == "Low"
    assert d["needs_review"] is True


def test_latency_under_2s(client):
    body = {"text": "sadak pe gaddha", "image_base64": _b64_gray()}
    _post(client, **body)  # warm-up
    t = time.perf_counter()
    _post(client, **body)
    assert time.perf_counter() - t < 2.0
