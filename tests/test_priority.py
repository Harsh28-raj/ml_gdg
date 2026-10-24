"""Tests for /ml/priority (rule-based, no models needed)."""
import pytest


def _post(client, **body):
    r = client.post("/ml/priority", json={"complaint_id": "P1", **body})
    assert r.status_code == 200
    return r.json()


def test_taar_gira_high(client):
    d = _post(client, text="Bijli ka taar gira hai", category="streetlight")
    assert d["priority"] == "High"
    assert "urgent keyword: taar" in d["reasons"]
    assert d["needs_review"] is False


def test_spec_example_with_duplicates(client):
    d = _post(client, text="Bijli ka taar gira hai", category="streetlight", duplicate_count=5)
    assert d["priority"] == "High" and d["priority_score"] >= 0.9
    assert "5 people reported this" in d["reasons"]


@pytest.mark.parametrize("text", ["Gali mein kachra pada hai", "kachra aage wali gali mein"])
def test_normal_kachra_low(client, text):
    d = _post(client, text=text, category="waste")
    assert d["priority"] in ("Low", "Medium")
    assert not any(r.startswith("urgent") for r in d["reasons"])  # "aage" must not match "aag"


def test_duplicate_count_boosts(client):
    base = _post(client, text="Gali mein kachra pada hai", category="waste")
    many = _post(client, text="Gali mein kachra pada hai", category="waste", duplicate_count=10)
    assert many["priority_score"] > base["priority_score"]
    assert many["priority"] == "Medium"


def test_duplicate_boost_capped(client):
    a = _post(client, text="kachra", category="waste", duplicate_count=50)
    b = _post(client, text="kachra", category="waste", duplicate_count=500)
    assert a["priority_score"] == b["priority_score"]


@pytest.mark.parametrize("text,kw", [
    ("Open manhole near the school", "open manhole"),
    ("Garbage burning, aag lagi hai", "aag"),
    ("bacche gir sakte hain gaddhe mein", "bacche"),
    ("Live wires hanging on road", "wire"),
])
def test_urgent_keywords(client, text, kw):
    d = _post(client, text=text, category="road")
    assert f"urgent keyword: {kw}" in d["reasons"]
    assert d["priority"] == "High"


def test_urgent_boost_capped(client):
    d = _post(client, text="taar current shock accident fire hospital school", category="other")
    assert d["priority_score"] == pytest.approx(0.2 + 0.5)


@pytest.mark.parametrize("body", [{}, {"text": "  ", "category": None}, {"category": "noise", "duplicate_count": -3}])
def test_missing_fields_no_crash(client, body):
    d = _post(client, **body)
    assert d["priority"] in ("High", "Medium", "Low")
    assert d["needs_review"] is True and d["reason"]
