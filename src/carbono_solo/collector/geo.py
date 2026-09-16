from __future__ import annotations

import math

# Lightweight conservative filter only; not a substitute for an official Brazil polygon.
BRAZIL_BBOX = {
    "min_lon": -74.0,
    "max_lon": -34.0,
    "min_lat": -34.0,
    "max_lat": 6.0,
}


def is_inside_brazil_bbox(latitude: float, longitude: float) -> bool:
    return (
        BRAZIL_BBOX["min_lat"] <= latitude <= BRAZIL_BBOX["max_lat"]
        and BRAZIL_BBOX["min_lon"] <= longitude <= BRAZIL_BBOX["max_lon"]
    )


def approx_duplicate_coord_key(latitude: float, longitude: float) -> str:
    return f"{latitude:.5f}|{longitude:.5f}"


def normalize_year(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        year_value = float(text)
    except (OverflowError, ValueError):
        return ""
    if not math.isfinite(year_value) or not year_value.is_integer():
        return ""
    year = int(year_value)
    if 1900 <= year <= 2100:
        return str(year)
    return ""
