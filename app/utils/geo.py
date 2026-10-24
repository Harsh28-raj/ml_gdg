"""Geo helpers."""
import math
from typing import Optional

from app.config import EARTH_RADIUS_M


def haversine_m(lat1: Optional[float], lng1: Optional[float],
                lat2: Optional[float], lng2: Optional[float]) -> Optional[float]:
    """Great-circle distance in meters; None if any coordinate is missing."""
    if None in (lat1, lng1, lat2, lng2):
        return None
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))
