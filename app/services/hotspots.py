"""Hotspot clustering: DBSCAN with haversine metric on (lat, lng) in radians."""
from collections import Counter

import numpy as np
from sklearn.cluster import DBSCAN

from app import config
from app.schemas import Cluster, HotspotsRequest, HotspotsResponse


def _valid(lat, lng) -> bool:
    return lat is not None and lng is not None and -90 <= lat <= 90 and -180 <= lng <= 180


def find_hotspots(req: HotspotsRequest) -> HotspotsResponse:
    """Cluster complaint locations; clusters sorted by size (largest = cluster 0). Never raises."""
    review: list[str] = []
    pts = [p for p in req.points if _valid(p.lat, p.lng)]
    if len(pts) < len(req.points):
        review.append(f"{len(req.points) - len(pts)} point(s) skipped: missing/invalid GPS")
    if not pts:
        return HotspotsResponse(needs_review=bool(review), reason="; ".join(review) or "no points")

    eps_m, min_samples = req.eps_m, req.min_samples
    if eps_m <= 0:
        eps_m = config.HOTSPOT_DEFAULT_EPS_M
        review.append(f"invalid eps_m; default {eps_m:.0f} m used")
    if min_samples < 1:
        min_samples = config.HOTSPOT_DEFAULT_MIN_SAMPLES
        review.append(f"invalid min_samples; default {min_samples} used")

    coords = np.radians([[p.lat, p.lng] for p in pts])
    labels = DBSCAN(eps=eps_m / config.EARTH_RADIUS_M, min_samples=min_samples,
                    metric="haversine", algorithm="ball_tree").fit_predict(coords)

    groups = [np.flatnonzero(labels == lab) for lab in set(labels) if lab != -1]
    groups.sort(key=len, reverse=True)
    clusters = []
    for cid, idx in enumerate(groups):
        members = [pts[i] for i in idx]
        cats = Counter(p.category for p in members if p.category)
        clusters.append(Cluster(
            cluster_id=cid,
            center_lat=round(float(np.mean([p.lat for p in members])), 6),
            center_lng=round(float(np.mean([p.lng for p in members])), 6),
            count=len(members),
            top_category=cats.most_common(1)[0][0] if cats else None,
            complaint_ids=[p.complaint_id for p in members],
        ))
    return HotspotsResponse(
        clusters=clusters,
        noise_count=int(np.sum(labels == -1)),
        needs_review=bool(review),
        reason="; ".join(review) or None,
    )
