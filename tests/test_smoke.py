"""Phase 0: server boots and every endpoint returns a contract-shaped response."""


def _common(body: dict) -> None:
    assert isinstance(body["needs_review"], bool)
    assert "reason" in body


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_openapi_docs(client):
    assert client.get("/docs").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    for p in ("/ml/classify", "/ml/duplicate", "/ml/priority", "/ml/verify", "/ml/hotspots"):
        assert p in paths


def test_classify_shape(client):
    b = client.post("/ml/classify", json={"complaint_id": "C1", "text": "kachra"}).json()
    _common(b)
    assert {"complaint_id", "category", "confidence", "confidence_label", "scores", "source"} <= b.keys()


def test_duplicate_shape(client):
    c = {"complaint_id": "C2", "lat": 28.67, "lng": 77.45, "category": "road"}
    b = client.post("/ml/duplicate", json={"complaint": c, "candidates": []}).json()
    _common(b)
    assert b["is_duplicate"] is False and b["group_id"] is None
    b = client.post("/ml/duplicate", json={"complaint": c, "candidates": [{**c, "complaint_id": "C1", "group_id": "G1"}]}).json()
    assert {"matched_complaint_id", "group_id", "similarity_score", "details", "top_matches"} <= b.keys()


def test_priority_shape(client):
    b = client.post("/ml/priority", json={"complaint_id": "C3", "text": "taar gira", "category": "streetlight", "duplicate_count": 5}).json()
    _common(b)
    assert b["priority"] in ("High", "Medium", "Low") and isinstance(b["reasons"], list)


def test_verify_shape(client):
    b = client.post("/ml/verify", json={"complaint_id": "C4"}).json()
    _common(b)
    assert b["status"] in ("Verified", "Rejected", "Suspicious")
    assert {"location_ok", "time_ok", "phash_distance", "same_place_score", "issue_removed"} <= b["checks"].keys()


def test_hotspots_shape(client):
    pts = [{"complaint_id": f"C{i}", "lat": 28.67, "lng": 77.45, "category": "waste"} for i in range(3)]
    b = client.post("/ml/hotspots", json={"points": pts, "eps_m": 200, "min_samples": 3}).json()
    _common(b)
    assert "clusters" in b and "noise_count" in b
