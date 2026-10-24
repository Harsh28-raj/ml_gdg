"""Hit every endpoint of a running ML service with the CLAUDE.md demo cases.

Usage: python scripts/demo.py [--base http://localhost:8001]
Image cases are skipped if the file is missing from sample_data/images/.
"""
import argparse
import json
import sys
from pathlib import Path

import requests

IMG = Path(__file__).resolve().parents[1] / "sample_data" / "images"
GPS = {"lat": 28.67, "lng": 77.45}
GPS_NEAR = {"lat": 28.6702, "lng": 77.4501}  # ~24 m


def has(*names: str) -> bool:
    missing = [n for n in names if not (IMG / n).exists()]
    if missing:
        print(f"   SKIP — missing sample_data/images/{', '.join(missing)}")
    return not missing


def img(name: str) -> str:
    return f"sample_data/images/{name}"


def call(base: str, path: str, body: dict, keys: list[str], expect: tuple[str, object] | None = None) -> None:
    r = requests.post(base + path, json=body, timeout=30)
    d = r.json()
    print("   " + json.dumps({k: d.get(k) for k in keys + ["needs_review", "reason"]}, ensure_ascii=False))
    if expect:
        k, v = expect
        print(f"   {'PASS' if d.get(k) == v else 'FAIL'}: {k} == {v!r}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows console
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8001")
    base = ap.parse_args().base

    print("GET /health\n   " + json.dumps(requests.get(base + "/health", timeout=10).json()))

    print("\n[1] Same pothole reported twice within 30 m → duplicate")
    a = {"complaint_id": "D1", "text": "Big pothole on the main road near the market", **GPS,
         "created_at": "2026-10-24T10:00:00", "category": "road"}
    b = {"complaint_id": "D0", "group_id": "G12", "text": "Large pothole near market road, very dangerous", **GPS_NEAR,
         "created_at": "2026-10-22T09:00:00", "category": "road"}
    if (IMG / "pothole1.jpg").exists() and (IMG / "pothole1_b.jpg").exists():
        a["image_url"], b["image_url"] = img("pothole1.jpg"), img("pothole1_b.jpg")
    call(base, "/ml/duplicate", {"complaint": a, "candidates": [b]},
         ["is_duplicate", "group_id", "similarity_score", "details"], ("is_duplicate", True))

    print("\n[2] Garbage photo, empty text → waste")
    if has("garbage1.jpg"):
        call(base, "/ml/classify", {"complaint_id": "C1", "text": "", "image_url": img("garbage1.jpg")},
             ["category", "confidence", "source"], ("category", "waste"))
    print("    text-only: 'Gali mein kachra pada hai'")
    call(base, "/ml/classify", {"complaint_id": "C2", "text": "Gali mein kachra pada hai"},
         ["category", "confidence", "source"], ("category", "waste"))

    print("\n[3] 'Bijli ka taar gira hai' → High priority")
    call(base, "/ml/priority", {"complaint_id": "P1", "text": "Bijli ka taar gira hai", "category": "streetlight",
                                "duplicate_count": 5}, ["priority", "priority_score", "reasons"], ("priority", "High"))

    meta = {"complaint_lat": 28.67, "complaint_lng": 77.45, "after_lat": 28.6701, "after_lng": 77.4502,
            "complaint_created_at": "2026-10-22T09:00:00", "after_taken_at": "2026-10-24T12:00:00"}
    keys = ["status", "verification_score", "checks", "reasons"]
    print("\n[4] Real before/after → Verified")
    if has("pothole1.jpg", "pothole1_fixed.jpg"):
        call(base, "/ml/verify", {"complaint_id": "V1", "category": "road", "before_image_url": img("pothole1.jpg"),
                                  "after_image_url": img("pothole1_fixed.jpg"), **meta}, keys, ("status", "Verified"))
    print("\n[5] Same photo as before and after → Rejected")
    if has("pothole1.jpg"):
        call(base, "/ml/verify", {"complaint_id": "V2", "category": "road", "before_image_url": img("pothole1.jpg"),
                                  "after_image_url": img("pothole1.jpg"), **meta}, keys, ("status", "Rejected"))
    print("\n[6] Clean road from a different place → Suspicious")
    if has("pothole1.jpg", "clean_road_other.jpg"):
        call(base, "/ml/verify", {"complaint_id": "V3", "category": "road", "before_image_url": img("pothole1.jpg"),
                                  "after_image_url": img("clean_road_other.jpg"), **meta}, keys, ("status", "Suspicious"))

    print("\n[+] Hotspots: 5 waste complaints in one block + 1 far away")
    pts = [{"complaint_id": f"H{i}", "lat": 28.67 + i * 0.0003, "lng": 77.45, "category": "waste"} for i in range(5)]
    pts.append({"complaint_id": "H9", "lat": 28.70, "lng": 77.50, "category": "road"})
    call(base, "/ml/hotspots", {"points": pts, "eps_m": 200, "min_samples": 3}, ["clusters", "noise_count"])


if __name__ == "__main__":
    main()
