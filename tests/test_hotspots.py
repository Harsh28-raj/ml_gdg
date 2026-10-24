"""Tests for /ml/hotspots (DBSCAN, no models needed)."""
import numpy as np

M_PER_DEG_LAT = 111_195  # meters per degree latitude


def _post(client, points, **kw):
    r = client.post("/ml/hotspots", json={"points": points, **kw})
    assert r.status_code == 200
    return r.json()


def _blob(prefix, lat, lng, n, spread_m, cat, seed=0):
    rng = np.random.default_rng(seed)
    off = rng.uniform(-spread_m, spread_m, (n, 2)) / M_PER_DEG_LAT
    return [{"complaint_id": f"{prefix}{i}", "lat": lat + dy, "lng": lng + dx / np.cos(np.radians(lat)), "category": cat}
            for i, (dy, dx) in enumerate(off)]


def test_two_clusters_and_noise(client):
    a = _blob("A", 28.670, 77.450, 7, 50, "waste")
    b = _blob("B", 28.700, 77.480, 4, 50, "road", seed=1)  # ~4.4 km away
    noise = [{"complaint_id": "N1", "lat": 28.60, "lng": 77.30, "category": "water"},
             {"complaint_id": "N2", "lat": 28.75, "lng": 77.60, "category": "water"}]
    d = _post(client, a + b + noise, eps_m=200, min_samples=3)
    assert len(d["clusters"]) == 2 and d["noise_count"] == 2
    c0, c1 = d["clusters"]
    assert (c0["cluster_id"], c0["count"], c0["top_category"]) == (0, 7, "waste")
    assert (c1["count"], c1["top_category"]) == (4, "road")
    assert set(c0["complaint_ids"]) == {p["complaint_id"] for p in a}
    assert abs(c0["center_lat"] - 28.670) < 0.001 and abs(c0["center_lng"] - 77.450) < 0.001


def test_eps_in_meters(client):
    pts = [{"complaint_id": f"P{i}", "lat": 28.67 + i * 150 / M_PER_DEG_LAT, "lng": 77.45} for i in range(3)]  # 150 m apart
    assert len(_post(client, pts, eps_m=200, min_samples=3)["clusters"]) == 1
    d = _post(client, pts, eps_m=100, min_samples=3)
    assert d["clusters"] == [] and d["noise_count"] == 3


def test_top_category_majority(client):
    pts = _blob("W", 28.67, 77.45, 3, 30, "waste") + _blob("R", 28.67, 77.45, 2, 30, "road", seed=2)
    d = _post(client, pts, eps_m=200, min_samples=3)
    assert d["clusters"][0]["top_category"] == "waste" and d["clusters"][0]["count"] == 5


def test_empty_points(client):
    d = _post(client, [])
    assert d["clusters"] == [] and d["noise_count"] == 0


def test_missing_gps_skipped(client):
    pts = _blob("A", 28.67, 77.45, 3, 30, "waste") + [{"complaint_id": "X"}, {"complaint_id": "Y", "lat": 999, "lng": 77}]
    d = _post(client, pts, eps_m=200, min_samples=3)
    assert len(d["clusters"]) == 1 and d["clusters"][0]["count"] == 3
    assert d["needs_review"] is True and "2 point(s) skipped" in d["reason"]


def test_bad_params_use_defaults(client):
    d = _post(client, _blob("A", 28.67, 77.45, 3, 30, "waste"), eps_m=0, min_samples=0)
    assert len(d["clusters"]) == 1 and d["needs_review"] is True


def test_defaults_when_params_omitted(client):
    d = _post(client, _blob("A", 28.67, 77.45, 3, 30, None))
    assert len(d["clusters"]) == 1 and d["clusters"][0]["top_category"] is None
