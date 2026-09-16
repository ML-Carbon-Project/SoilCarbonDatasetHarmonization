from __future__ import annotations

import math
import re
from urllib.parse import urlencode

from carbono_solo.collector.geo import is_inside_brazil_bbox
from carbono_solo.collector.io import fetch_json
from carbono_solo.collector.models import (
    CATALOG_ONLY,
    FAILED,
    IMPLEMENTED,
    InventoryRecord,
    SourceResult,
    TargetRecord,
)


SOURCE_ID = "wosis_brazil"
SOURCE_NAME = "WoSIS latest Brazil subset"
DATASET_ID = "wosis_latest_orgc"
WFS_BASE_URL = "https://maps.isric.org/mapserv?map=/map/wosis_latest.map"
WOSIS_ORGC_LAYER = "wosis_latest_orgc"
MIN_RETRY_PAGE_SIZE = 1
MAX_RETRY_PAGE_SIZE = 1000

# WFS 2.0 with EPSG:4326 expects axis order latitude, longitude.
BRAZIL_WFS_BBOX = "-34,-74,6,-34,urn:ogc:def:crs:EPSG::4326"


def collect_wosis_brazil(
    *,
    retrieved_at: str,
    use_network: bool = True,
    page_size: int = 5000,
) -> SourceResult:
    if not use_network:
        return SourceResult(inventory=[], targets=[])

    try:
        targets: list[TargetRecord] = []
        start_index = 0
        current_page_size = page_size
        while True:
            try:
                payload = fetch_json(
                    _wfs_url(start_index=start_index, page_size=current_page_size)
                )
            except RuntimeError:
                retry_page_size = max(
                    MIN_RETRY_PAGE_SIZE,
                    min(MAX_RETRY_PAGE_SIZE, current_page_size // 2),
                )
                if retry_page_size >= current_page_size:
                    raise
                current_page_size = retry_page_size
                continue

            features = payload.get("features", []) if isinstance(payload, dict) else []
            if not isinstance(features, list):
                raise ValueError("WoSIS WFS response has no feature list.")
            targets.extend(targets_from_feature_collection(payload))
            if len(features) < current_page_size:
                break
            start_index += current_page_size

        return SourceResult(
            inventory=[
                _inventory_record(
                    retrieved_at=retrieved_at,
                    status=IMPLEMENTED if targets else CATALOG_ONLY,
                    notes=(
                        f"Parsed {len(targets)} target rows from WFS layer "
                        f"{WOSIS_ORGC_LAYER}, filtered to Brazil."
                    ),
                )
            ],
            targets=targets,
        )
    except Exception as exc:
        return SourceResult(
            inventory=[
                _inventory_record(
                    retrieved_at=retrieved_at,
                    status=FAILED,
                    notes=f"Ingestion error: {exc}",
                )
            ],
            targets=[],
        )


def targets_from_feature_collection(payload: dict) -> list[TargetRecord]:
    features = payload.get("features", [])
    if not isinstance(features, list):
        return []

    records: list[TargetRecord] = []
    for feature in features:
        if not isinstance(feature, dict):
            continue
        properties = feature.get("properties", {})
        geometry = feature.get("geometry", {})
        if not isinstance(properties, dict) or not isinstance(geometry, dict):
            continue
        if str(properties.get("country_name", "")).strip().lower() != "brazil":
            continue

        coordinates = geometry.get("coordinates", [])
        if not isinstance(coordinates, list) or len(coordinates) < 2:
            continue
        longitude = _to_float(coordinates[0])
        latitude = _to_float(coordinates[1])
        value = _to_float(properties.get("value_avg"))
        upper_depth = _to_float(properties.get("upper_depth"))
        lower_depth = _to_float(properties.get("lower_depth"))
        if None in {latitude, longitude, value, upper_depth, lower_depth}:
            continue
        if not is_inside_brazil_bbox(latitude=latitude, longitude=longitude):
            continue

        profile_id = _text(properties.get("profile_id"))
        layer_id = _text(properties.get("layer_id"))
        source_dataset = _text(properties.get("dataset_id")) or "wosis"
        sample_date, sample_year = _sample_period(_text(properties.get("date")))
        depth_top = _format_number(upper_depth)
        depth_bottom = _format_number(lower_depth)
        target_value = _format_number(value)
        observation_id = f"{source_dataset}:{profile_id}:{layer_id}"
        quality_flag = (
            "organic_surface"
            if str(properties.get("organic_surface", "")).lower() == "true"
            else "ok"
        )
        unified_id = (
            f"{SOURCE_ID}:{DATASET_ID}:{observation_id}:"
            f"{depth_top}-{depth_bottom}:carbon_content"
        )

        records.append(
            TargetRecord(
                unified_observation_id=unified_id,
                source_id=SOURCE_ID,
                dataset_id=DATASET_ID,
                source_observation_id=observation_id,
                latitude=latitude,
                longitude=longitude,
                coord_x=longitude,
                coord_y=latitude,
                crs="EPSG:4326",
                country="Brazil",
                state="",
                sample_date=sample_date,
                sample_year=sample_year,
                sample_period_start=sample_year,
                sample_period_end=sample_year,
                depth_top_cm=depth_top,
                depth_bottom_cm=depth_bottom,
                layer_thickness_cm=_format_number(lower_depth - upper_depth),
                carbon_content_g_kg=target_value,
                carbon_stock_kg_m2="",
                carbon_stock_mg_ha="",
                bulk_density_g_cm3="",
                coarse_fragments_fraction="",
                carbon_stock_00_05="",
                carbon_stock_05_15="",
                carbon_stock_15_30="",
                carbon_stock_30_60="",
                carbon_stock_60_100="",
                target_kind="carbon_content",
                target_value=target_value,
                target_unit="g/kg",
                derivation_method="reported_by_wosis_latest_orgc",
                quality_flag=quality_flag,
                duplicate_group_id="",
                duplicate_status="active",
                duplicate_resolution="unique",
            )
        )
    return records


def _wfs_url(*, start_index: int, page_size: int) -> str:
    params = {
        "SERVICE": "WFS",
        "VERSION": "2.0.0",
        "REQUEST": "GetFeature",
        "TYPENAMES": WOSIS_ORGC_LAYER,
        "OUTPUTFORMAT": "geojson",
        "COUNT": str(page_size),
        "STARTINDEX": str(start_index),
        "BBOX": BRAZIL_WFS_BBOX,
    }
    return f"{WFS_BASE_URL}&{urlencode(params)}"


def _inventory_record(*, retrieved_at: str, status: str, notes: str) -> InventoryRecord:
    return InventoryRecord(
        source_id=SOURCE_ID,
        source_name=SOURCE_NAME,
        dataset_id=DATASET_ID,
        title="WoSIS latest Organic carbon filtered to Brazil",
        url=(
            "https://maps.isric.org/mapserv?map=/map/wosis_latest.map"
            "&SERVICE=WFS&REQUEST=GetFeature&TYPENAMES=wosis_latest_orgc"
        ),
        country_scope="Brazil",
        access_type="ogc_wfs_geojson",
        brazil_filter="wfs_bbox_and_country_name_brazil",
        candidate_target_fields="organic carbon, layer depth, coordinates, sample date",
        target_fit_score="high",
        implementation_status=status,
        license="feature_level_license",
        notes=notes,
        retrieved_at=retrieved_at,
    )


def _sample_period(raw_date: str) -> tuple[str, str]:
    if re.match(r"^\d{4}-\d{2}-\d{2}$", raw_date):
        return raw_date, raw_date[:4]
    year_match = re.search(r"(19|20)\d{2}", raw_date)
    return "", year_match.group(0) if year_match else ""


def _to_float(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _text(value: object) -> str:
    return str(value or "").strip()
