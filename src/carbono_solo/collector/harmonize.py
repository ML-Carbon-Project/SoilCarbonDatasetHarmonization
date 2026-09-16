from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Iterable, Mapping

from carbono_solo.collector.geo import is_inside_brazil_bbox, normalize_year
from carbono_solo.collector.io import write_csv
from carbono_solo.collector.lab_methods import NOT_REPORTED


TRACE_FIELDS = [
    "SOURCE_ID",
    "DATASET_ID",
]

BASE_FIELDS_BEFORE_DEPTH = [
    *TRACE_FIELDS,
    "LATITUDE",
    "LONGITUDE",
    "CRS",
    "COUNTRY",
    "STATE",
    "SAMPLE_DATE",
    "SAMPLE_YEAR",
    "DEPTH_TOP_CM",
    "DEPTH_BOTTOM_CM",
    "LAYER_THICKNESS_CM",
]

BASE_FIELDS_AFTER_DEPTH = [
    "CARBON_CONTENT_G_KG",
    "CARBON_ANALYSIS_METHOD",
    "BULK_DENSITY_G_CM3",
    "COARSE_FRAGMENTS_FRACTION",
    "CARBON_STOCK_KG_M2",
    "CARBON_STOCK_Mg_ha",
    "TARGET_VALUE",
    "TARGET_UNIT",
    "STOCK_CALCULATION_METHOD",
    "ORIGINAL_TARGET_KIND",
    "ORIGINAL_TARGET_VALUE",
    "ORIGINAL_TARGET_UNIT",
    "QUALITY_FLAG",
    "DUPLICATE_GROUP_ID",
    "DUPLICATE_STATUS",
    "DUPLICATE_RESOLUTION",
]

STATE_BBOXES = [
    ("AC", -11.2, -7.0, -74.1, -66.5),
    ("AL", -10.6, -8.7, -38.3, -35.1),
    ("AP", -1.4, 4.6, -55.0, -49.7),
    ("AM", -10.0, 2.4, -74.1, -56.0),
    ("BA", -18.4, -8.5, -46.8, -37.2),
    ("CE", -7.9, -2.7, -41.5, -37.1),
    ("DF", -16.1, -15.4, -48.3, -47.3),
    ("ES", -21.4, -17.8, -41.9, -39.6),
    ("GO", -19.6, -12.4, -53.4, -45.9),
    ("MA", -10.3, -1.0, -48.9, -41.7),
    ("MT", -18.2, -7.0, -61.8, -50.2),
    ("MS", -24.2, -17.0, -58.3, -50.9),
    ("MG", -22.9, -14.0, -51.1, -39.8),
    ("PA", -10.0, 2.8, -59.0, -46.0),
    ("PB", -8.4, -6.0, -38.9, -34.7),
    ("PR", -26.8, -22.4, -54.7, -48.0),
    ("PE", -9.6, -7.2, -41.4, -34.8),
    ("PI", -11.0, -2.7, -46.0, -40.3),
    ("RJ", -23.4, -20.7, -44.9, -40.9),
    ("RN", -7.0, -4.8, -38.7, -34.9),
    ("RS", -33.8, -27.0, -57.8, -49.6),
    ("RO", -13.8, -7.8, -67.0, -59.7),
    ("RR", -1.7, 5.4, -64.9, -58.8),
    ("SC", -29.4, -25.8, -53.9, -48.3),
    ("SP", -25.4, -19.7, -53.2, -44.0),
    ("SE", -11.6, -9.5, -38.4, -36.3),
    ("TO", -13.7, -5.0, -50.8, -45.7),
]

ORGANIC_MATTER_TO_CARBON_FRACTION = 0.58
BULK_DENSITY_MIN_G_CM3 = 0.02
BULK_DENSITY_MAX_G_CM3 = 2.2
CARBON_CONTENT_MAX_G_KG = 1000.0

HARMONIZATION_AUDIT_FIELDS = [
    "UNIFIED_OBSERVATION_ID",
    "SOURCE_ID",
    "DATASET_ID",
    "SOURCE_OBSERVATION_ID",
    "LATITUDE",
    "LONGITUDE",
    "ORIGINAL_DEPTH_TOP_CM",
    "ORIGINAL_DEPTH_BOTTOM_CM",
    "DEPTH_TOP_CM",
    "DEPTH_BOTTOM_CM",
    "BULK_DENSITY_G_CM3",
    "CARBON_CONTENT_G_KG",
    "OUTPUT_STATUS",
    "EXCLUSION_REASON",
]

ORGANIC_SURFACE_DENSITY_EXCLUSION = (
    "organic_surface_layer_with_bulk_density_below_mineral_limit"
)


def harmonize_csv(
    input_path: str | Path,
    output_path: str | Path,
    audit_path: str | Path | None = None,
) -> dict[str, int]:
    rows = _read_csv(input_path)
    fields, harmonized_rows, audit_rows = harmonize_target_rows_with_audit(rows)
    write_csv(output_path, fields, harmonized_rows)
    if audit_path is not None:
        write_csv(audit_path, HARMONIZATION_AUDIT_FIELDS, audit_rows)
    return {
        "input_rows": len(rows),
        "output_rows": len(harmonized_rows),
        "excluded_organic_surface_density_rows": len(audit_rows),
        "dynamic_depth_columns": len(
            [
                field
                for field in fields
                if field.startswith("DEPTH_")
                and field not in {"DEPTH_TOP_CM", "DEPTH_BOTTOM_CM"}
            ]
        ),
        "stock_targets": sum(1 for row in harmonized_rows if row["TARGET_VALUE"]),
    }


def harmonize_target_rows(
    raw_rows: Iterable[Mapping[str, object]],
) -> tuple[list[str], list[dict[str, str]]]:
    fields, harmonized_rows, _audit_rows = harmonize_target_rows_with_audit(raw_rows)
    return fields, harmonized_rows


def harmonize_target_rows_with_audit(
    raw_rows: Iterable[Mapping[str, object]],
) -> tuple[list[str], list[dict[str, str]], list[dict[str, str]]]:
    rows = [dict(row) for row in raw_rows]
    mean_reference = _mean_reference(rows)
    fields = [*BASE_FIELDS_BEFORE_DEPTH, *BASE_FIELDS_AFTER_DEPTH]
    harmonized_rows = []
    audit_rows = []
    for row in rows:
        harmonized = _harmonize_row(row, mean_reference)
        exclusion_reason = _exclusion_reason(row, harmonized)
        if exclusion_reason:
            audit_rows.append(_audit_row(row, harmonized, exclusion_reason))
        elif harmonized["STATE"]:
            harmonized_rows.append(harmonized)
    return fields, harmonized_rows, audit_rows


def _exclusion_reason(
    raw_row: Mapping[str, object],
    harmonized_row: Mapping[str, str],
) -> str:
    depth_top = _to_float(harmonized_row.get("DEPTH_TOP_CM", ""))
    depth_bottom = _to_float(harmonized_row.get("DEPTH_BOTTOM_CM", ""))
    raw_bulk_density = _to_float(_value(raw_row, "bulk_density_g_cm3"))
    if (
        depth_top is not None
        and depth_bottom is not None
        and depth_top < 0
        and depth_bottom <= 0
        and raw_bulk_density is not None
        and 0 < raw_bulk_density < BULK_DENSITY_MIN_G_CM3
    ):
        return ORGANIC_SURFACE_DENSITY_EXCLUSION
    return ""


def _audit_row(
    raw_row: Mapping[str, object],
    harmonized_row: Mapping[str, str],
    reason: str,
) -> dict[str, str]:
    return {
        "UNIFIED_OBSERVATION_ID": _value(raw_row, "unified_observation_id"),
        "SOURCE_ID": _value(raw_row, "source_id"),
        "DATASET_ID": _value(raw_row, "dataset_id"),
        "SOURCE_OBSERVATION_ID": _value(raw_row, "source_observation_id"),
        "LATITUDE": harmonized_row["LATITUDE"],
        "LONGITUDE": harmonized_row["LONGITUDE"],
        "ORIGINAL_DEPTH_TOP_CM": _value(raw_row, "depth_top_cm"),
        "ORIGINAL_DEPTH_BOTTOM_CM": _value(raw_row, "depth_bottom_cm"),
        "DEPTH_TOP_CM": harmonized_row["DEPTH_TOP_CM"],
        "DEPTH_BOTTOM_CM": harmonized_row["DEPTH_BOTTOM_CM"],
        "BULK_DENSITY_G_CM3": _value(raw_row, "bulk_density_g_cm3"),
        "CARBON_CONTENT_G_KG": harmonized_row["CARBON_CONTENT_G_KG"],
        "OUTPUT_STATUS": "excluded",
        "EXCLUSION_REASON": reason,
    }


def _read_csv(input_path: str | Path) -> list[dict[str, str]]:
    with Path(input_path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=";"))


def _harmonize_row(
    row: Mapping[str, object],
    mean_reference: dict[str, dict[str, float]],
) -> dict[str, str]:
    latitude, longitude, _coordinate_validation = _coordinates(row)
    country, _country_method = _country(row, latitude, longitude)
    state, _state_method = _state(row, latitude, longitude, country)
    depth_top = _to_float(_value(row, "depth_top_cm"))
    depth_bottom = _to_float(_value(row, "depth_bottom_cm"))
    depth_top, depth_bottom = _ordered_depths(depth_top, depth_bottom)
    layer_thickness = _layer_thickness(row, depth_top, depth_bottom)
    carbon_content, carbon_measure_kind = _carbon_content(row)
    bulk_density, bulk_density_source = _bulk_density(row)
    coarse_fraction = _coarse_fraction(row)
    if bulk_density is None:
        bulk_density, bulk_density_source = _estimate_mean_value(
            mean_reference,
            "bulk_density",
            state=state,
            depth_top=depth_top,
            depth_bottom=depth_bottom,
            carbon_content=carbon_content,
        )
    if coarse_fraction is None:
        coarse_fraction, _coarse_fraction_source = _estimate_mean_value(
            mean_reference,
            "coarse_fraction",
            state=state,
            depth_top=depth_top,
            depth_bottom=depth_bottom,
        )
    stock = _stock_values(
        row,
        carbon_content=carbon_content,
        carbon_measure_kind=carbon_measure_kind,
        bulk_density=bulk_density,
        bulk_density_source=bulk_density_source,
        layer_thickness=layer_thickness,
        coarse_fraction=coarse_fraction,
    )
    if carbon_content is None:
        back_calculated_content = _back_calculate_carbon_content(
            stock_mg_ha=_to_float(stock["mg_ha"]),
            bulk_density=bulk_density,
            layer_thickness=layer_thickness,
            coarse_fraction=coarse_fraction,
        )
        if back_calculated_content is not None:
            carbon_content = back_calculated_content
            stock["method"] = _back_calculation_method(
                str(stock["method"]),
                bulk_density_source,
            )

    return {
        "SOURCE_ID": _value(row, "source_id"),
        "DATASET_ID": _value(row, "dataset_id"),
        "LATITUDE": _format_number(latitude),
        "LONGITUDE": _format_number(longitude),
        "CRS": _value(row, "crs") or "EPSG:4326",
        "COUNTRY": country,
        "STATE": state,
        "SAMPLE_DATE": _value(row, "sample_date"),
        "SAMPLE_YEAR": _sample_year(row),
        "DEPTH_TOP_CM": _format_number(depth_top),
        "DEPTH_BOTTOM_CM": _format_number(depth_bottom),
        "LAYER_THICKNESS_CM": _format_number(layer_thickness),
        "CARBON_CONTENT_G_KG": _format_number(carbon_content),
        "CARBON_ANALYSIS_METHOD": (
            _value(row, "carbon_analysis_method") or NOT_REPORTED
        ),
        "BULK_DENSITY_G_CM3": _format_number(bulk_density),
        "COARSE_FRAGMENTS_FRACTION": _format_number(coarse_fraction),
        "CARBON_STOCK_KG_M2": _format_number(stock["kg_m2"]),
        "CARBON_STOCK_Mg_ha": _format_number(stock["mg_ha"]),
        "TARGET_VALUE": _format_number(stock["mg_ha"]),
        "TARGET_UNIT": "Mg/ha",
        "STOCK_CALCULATION_METHOD": stock["method"],
        "ORIGINAL_TARGET_KIND": _value(row, "target_kind"),
        "ORIGINAL_TARGET_VALUE": _value(row, "target_value"),
        "ORIGINAL_TARGET_UNIT": _value(row, "target_unit"),
        "QUALITY_FLAG": _value(row, "quality_flag"),
        "DUPLICATE_GROUP_ID": _value(row, "duplicate_group_id"),
        "DUPLICATE_STATUS": _value(row, "duplicate_status"),
        "DUPLICATE_RESOLUTION": _value(row, "duplicate_resolution"),
    }


def _coordinates(row: Mapping[str, object]) -> tuple[float | None, float | None, str]:
    raw_latitude = _to_float(_value(row, "latitude"))
    raw_longitude = _to_float(_value(row, "longitude"))
    coord_x = _to_float(_value(row, "coord_x"))
    coord_y = _to_float(_value(row, "coord_y"))

    latitude = raw_latitude
    if latitude is None and _is_valid_latitude(coord_y):
        latitude = coord_y
    if latitude is None and _is_valid_latitude(coord_x):
        latitude = coord_x

    longitude = raw_longitude
    if longitude is None and _is_valid_longitude(coord_x):
        longitude = coord_x
    if longitude is None and _is_valid_longitude(coord_y):
        longitude = coord_y

    if latitude is None or longitude is None:
        return latitude, longitude, "missing_coordinates"

    if coord_x is None or coord_y is None:
        return latitude, longitude, "missing_coord_validation_inputs"
    if _close(coord_x, longitude) and _close(coord_y, latitude):
        return latitude, longitude, "coord_x_longitude_coord_y_latitude"
    if _close(coord_x, latitude) and _close(coord_y, longitude):
        return latitude, longitude, "coord_x_latitude_coord_y_longitude"
    return latitude, longitude, "coord_mismatch"


def _country(
    row: Mapping[str, object],
    latitude: float | None,
    longitude: float | None,
) -> tuple[str, str]:
    country = _value(row, "country")
    if country:
        return country, "existing"
    if latitude is not None and longitude is not None and is_inside_brazil_bbox(
        latitude=latitude,
        longitude=longitude,
    ):
        return "Brazil", "coordinate_bbox"
    return "", "unresolved"


def _state(
    row: Mapping[str, object],
    latitude: float | None,
    longitude: float | None,
    country: str,
) -> tuple[str, str]:
    state = _value(row, "state").upper()
    if state:
        return state, "existing"
    if country.lower() != "brazil" or latitude is None or longitude is None:
        return "", "unresolved"
    return _state_from_coordinate(latitude, longitude)


def _state_from_coordinate(latitude: float, longitude: float) -> tuple[str, str]:
    matches = [
        (state, min_lat, max_lat, min_lon, max_lon)
        for state, min_lat, max_lat, min_lon, max_lon in STATE_BBOXES
        if min_lat <= latitude <= max_lat and min_lon <= longitude <= max_lon
    ]
    if not matches:
        return "", "unresolved"
    if len(matches) == 1:
        return matches[0][0], "coordinate_bbox"

    def distance_to_center(item: tuple[str, float, float, float, float]) -> float:
        _state, min_lat, max_lat, min_lon, max_lon = item
        center_lat = (min_lat + max_lat) / 2
        center_lon = (min_lon + max_lon) / 2
        return math.hypot(latitude - center_lat, longitude - center_lon)

    return min(matches, key=distance_to_center)[0], "coordinate_bbox_ambiguous"


def _sample_year(row: Mapping[str, object]) -> str:
    year = normalize_year(_value(row, "sample_year"))
    if year:
        return year
    match = re.search(r"(19|20)\d{2}", _value(row, "sample_date"))
    return match.group(0) if match else ""


def _layer_thickness(
    row: Mapping[str, object],
    depth_top: float | None,
    depth_bottom: float | None,
) -> float | None:
    if depth_top is not None and depth_bottom is not None:
        computed = depth_bottom - depth_top
        if computed >= 0:
            return computed
    thickness = _to_float(_value(row, "layer_thickness_cm"))
    if thickness is not None and thickness > 0:
        return thickness
    return None


def _ordered_depths(
    depth_top: float | None,
    depth_bottom: float | None,
) -> tuple[float | None, float | None]:
    if depth_top is None or depth_bottom is None:
        return depth_top, depth_bottom
    if depth_bottom < depth_top:
        return depth_bottom, depth_top
    return depth_top, depth_bottom


def _mean_reference(rows: list[Mapping[str, object]]) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, list[float]]] = {
        "bulk_density": {},
        "coarse_fraction": {},
    }
    for row in rows:
        latitude, longitude, _coordinate_validation = _coordinates(row)
        country, _country_method = _country(row, latitude, longitude)
        state, _state_method = _state(row, latitude, longitude, country)
        if not state:
            continue
        depth_top = _to_float(_value(row, "depth_top_cm"))
        depth_bottom = _to_float(_value(row, "depth_bottom_cm"))
        depth_top, depth_bottom = _ordered_depths(depth_top, depth_bottom)
        depth_bucket = _depth_bucket(depth_top, depth_bottom)
        carbon_content, _carbon_measure_kind = _carbon_content(row)
        carbon_bin = _carbon_bin(carbon_content)

        bulk_density, _bulk_density_source = _bulk_density(row)
        if (
            bulk_density is not None
            and BULK_DENSITY_MIN_G_CM3 <= bulk_density <= BULK_DENSITY_MAX_G_CM3
        ):
            _append_reference_value(
                values["bulk_density"],
                state=state,
                depth_bucket=depth_bucket,
                carbon_bin=carbon_bin,
                value=bulk_density,
            )

        coarse_fraction = _coarse_fraction(row)
        if coarse_fraction is not None and 0 <= coarse_fraction <= 1:
            _append_reference_value(
                values["coarse_fraction"],
                state=state,
                depth_bucket=depth_bucket,
                carbon_bin="",
                value=coarse_fraction,
            )

    return {
        kind: {
            key: sum(raw_values) / len(raw_values)
            for key, raw_values in kind_values.items()
            if raw_values
        }
        for kind, kind_values in values.items()
    }


def _append_reference_value(
    values: dict[str, list[float]],
    *,
    state: str,
    depth_bucket: str,
    carbon_bin: str,
    value: float,
) -> None:
    if carbon_bin:
        values.setdefault(f"carbon_bin:{carbon_bin}", []).append(value)
    if depth_bucket:
        values.setdefault(f"state_depth:{state}:{depth_bucket}", []).append(value)
        values.setdefault(f"depth:{depth_bucket}", []).append(value)
    values.setdefault(f"state:{state}", []).append(value)
    values.setdefault("global", []).append(value)


def _estimate_mean_value(
    mean_reference: dict[str, dict[str, float]],
    kind: str,
    *,
    state: str,
    depth_top: float | None,
    depth_bottom: float | None,
    carbon_content: float | None = None,
) -> tuple[float | None, str]:
    values = mean_reference.get(kind, {})
    depth_bucket = _depth_bucket(depth_top, depth_bottom)
    carbon_bin = _carbon_bin(carbon_content)
    candidates = []
    if kind == "bulk_density" and carbon_content is not None and carbon_content >= 60 and carbon_bin:
        candidates.append((f"carbon_bin:{carbon_bin}", "carbon_bin"))
    if state and depth_bucket:
        candidates.append((f"state_depth:{state}:{depth_bucket}", "state_depth"))
    if depth_bucket:
        candidates.append((f"depth:{depth_bucket}", "depth"))
    if state:
        candidates.append((f"state:{state}", "state"))
    candidates.append(("global", "global"))

    for key, source in candidates:
        value = values.get(key)
        if value is not None:
            return value, source
    return None, "missing"


def _depth_bucket(depth_top: float | None, depth_bottom: float | None) -> str:
    if depth_top is None or depth_bottom is None:
        return ""
    midpoint = (depth_top + depth_bottom) / 2
    if midpoint < 30:
        return "00_30"
    if midpoint < 60:
        return "30_60"
    if midpoint < 100:
        return "60_100"
    return "100_plus"


def _carbon_bin(carbon_content: float | None) -> str:
    if carbon_content is None:
        return ""
    if carbon_content < 10:
        return "000_010"
    if carbon_content < 30:
        return "010_030"
    if carbon_content < 60:
        return "030_060"
    if carbon_content < 100:
        return "060_100"
    if carbon_content < 200:
        return "100_200"
    return "200_plus"


def _carbon_content(row: Mapping[str, object]) -> tuple[float | None, str]:
    target_kind = _value(row, "target_kind")
    carbon_content = _to_float(_value(row, "carbon_content_g_kg"))
    if carbon_content is not None:
        measure_kind = (
            "carbon_fraction"
            if target_kind.startswith("carbon_fraction:")
            else "carbon_content"
        )
        return carbon_content, measure_kind
    if _value(row, "target_unit").lower() != "g/kg":
        return None, ""
    if target_kind in {"carbon_content", "total_carbon_content"}:
        return _to_float(_value(row, "target_value")), "carbon_content"
    if target_kind.startswith("carbon_fraction:"):
        return _to_float(_value(row, "target_value")), "carbon_fraction"
    return None, ""


def _coarse_fraction(row: Mapping[str, object]) -> float | None:
    raw_value = _to_float(_value(row, "coarse_fragments_fraction"))
    if raw_value is None:
        return None
    if raw_value > 1:
        return raw_value / 100
    return raw_value


def _bulk_density(row: Mapping[str, object]) -> tuple[float | None, str]:
    raw_value = _to_float(_value(row, "bulk_density_g_cm3"))
    if raw_value is None or raw_value <= 0:
        return None, "missing"
    if raw_value > 10:
        scaled_value = raw_value / 100
        if BULK_DENSITY_MIN_G_CM3 <= scaled_value <= BULK_DENSITY_MAX_G_CM3:
            return scaled_value, "observed_scaled"
        return None, "invalid"
    if not BULK_DENSITY_MIN_G_CM3 <= raw_value <= BULK_DENSITY_MAX_G_CM3:
        return None, "invalid"
    return raw_value, "observed"


def _stock_values(
    row: Mapping[str, object],
    *,
    carbon_content: float | None,
    carbon_measure_kind: str,
    bulk_density: float | None,
    bulk_density_source: str,
    layer_thickness: float | None,
    coarse_fraction: float | None,
) -> dict[str, float | str | None]:
    stock_kg_m2 = _to_float(_value(row, "carbon_stock_kg_m2"))
    if stock_kg_m2 is not None:
        return {
            "kg_m2": stock_kg_m2,
            "mg_ha": stock_kg_m2 * 10,
            "method": "converted_from_kg_m2",
        }

    stock_mg_ha = _to_float(_value(row, "carbon_stock_mg_ha"))
    if stock_mg_ha is not None:
        return {
            "kg_m2": stock_mg_ha / 10,
            "mg_ha": stock_mg_ha,
            "method": "existing_mg_ha",
        }

    target_stock = _stock_from_original_target(row)
    if target_stock["mg_ha"] is not None:
        return target_stock

    organic_matter_stock = _stock_from_organic_matter_proxy(
        row,
        layer_thickness=layer_thickness,
        coarse_fraction=coarse_fraction,
    )
    if organic_matter_stock["mg_ha"] is not None:
        return organic_matter_stock

    if carbon_content is not None and bulk_density is not None and layer_thickness is not None:
        coarse = coarse_fraction or 0.0
        stock_kg_m2 = carbon_content * bulk_density * layer_thickness * (1 - coarse) / 100
        measure_name = (
            "carbon_fraction"
            if carbon_measure_kind == "carbon_fraction"
            else "carbon_content"
        )
        if bulk_density_source == "observed":
            method = f"computed_from_{measure_name}_bulk_density_depth"
        elif bulk_density_source == "observed_scaled":
            method = f"computed_from_{measure_name}_scaled_bulk_density_depth"
        else:
            method = f"estimated_from_{measure_name}_mean_bulk_density_{bulk_density_source}"
        return {
            "kg_m2": stock_kg_m2,
            "mg_ha": stock_kg_m2 * 10,
            "method": method,
        }

    return {
        "kg_m2": None,
        "mg_ha": None,
        "method": "insufficient_inputs",
    }


def _back_calculate_carbon_content(
    *,
    stock_mg_ha: float | None,
    bulk_density: float | None,
    layer_thickness: float | None,
    coarse_fraction: float | None,
) -> float | None:
    if (
        stock_mg_ha is None
        or stock_mg_ha < 0
        or bulk_density is None
        or bulk_density <= 0
        or layer_thickness is None
        or layer_thickness <= 0
    ):
        return None
    coarse = coarse_fraction if coarse_fraction is not None else 0.0
    if not 0 <= coarse < 1:
        return None
    denominator = bulk_density * layer_thickness * (1 - coarse)
    if denominator <= 0:
        return None
    carbon_content = stock_mg_ha * 10 / denominator
    if not math.isfinite(carbon_content):
        return None
    if not 0 <= carbon_content <= CARBON_CONTENT_MAX_G_KG:
        return None
    return carbon_content


def _back_calculation_method(stock_method: str, bulk_density_source: str) -> str:
    density_label = {
        "observed": "observed_bulk_density",
        "observed_scaled": "scaled_observed_bulk_density",
    }.get(
        bulk_density_source,
        f"estimated_bulk_density_{bulk_density_source}",
    )
    return (
        f"{stock_method}_with_back_calculated_carbon_content_{density_label}"
    )


def _stock_from_original_target(row: Mapping[str, object]) -> dict[str, float | str | None]:
    target_kind = _value(row, "target_kind")
    target_value = _to_float(_value(row, "target_value"))
    target_unit = _value(row, "target_unit").lower().replace(" ", "")
    if target_value is None or not target_kind.startswith("carbon_stock"):
        return {"kg_m2": None, "mg_ha": None, "method": "not_stock_target"}
    if target_unit in {"mg/ha", "t/ha", "ton/ha", "mg.ha-1"}:
        return {
            "kg_m2": target_value / 10,
            "mg_ha": target_value,
            "method": "existing_target_mg_ha",
        }
    if target_unit in {"kg/m2", "kg/m²", "kgm-2"}:
        return {
            "kg_m2": target_value,
            "mg_ha": target_value * 10,
            "method": "converted_target_kg_m2",
        }
    if target_unit in {"g/m2", "g/m²", "gm-2"}:
        stock_kg_m2 = target_value / 1000
        return {
            "kg_m2": stock_kg_m2,
            "mg_ha": stock_kg_m2 * 10,
            "method": "converted_target_g_m2",
        }
    return {"kg_m2": None, "mg_ha": None, "method": "unsupported_stock_unit"}


def _stock_from_organic_matter_proxy(
    row: Mapping[str, object],
    *,
    layer_thickness: float | None,
    coarse_fraction: float | None,
) -> dict[str, float | str | None]:
    target_kind = _value(row, "target_kind")
    target_value = _to_float(_value(row, "target_value"))
    target_unit = _value(row, "target_unit").lower().replace(" ", "")
    if (
        target_kind != "organic_matter_proxy"
        or target_value is None
        or layer_thickness is None
        or target_unit not in {"g/dm3", "g/dm³"}
    ):
        return {"kg_m2": None, "mg_ha": None, "method": "not_organic_matter_proxy"}

    coarse = coarse_fraction or 0.0
    carbon_kg_m3 = target_value * ORGANIC_MATTER_TO_CARBON_FRACTION
    stock_kg_m2 = carbon_kg_m3 * (layer_thickness / 100) * (1 - coarse)
    return {
        "kg_m2": stock_kg_m2,
        "mg_ha": stock_kg_m2 * 10,
        "method": "estimated_from_organic_matter_proxy_g_dm3_van_bemmelen_0_58",
    }


def _is_valid_latitude(value: float | None) -> bool:
    return value is not None and -90 <= value <= 90


def _is_valid_longitude(value: float | None) -> bool:
    return value is not None and -180 <= value <= 180


def _close(left: float, right: float) -> bool:
    return abs(left - right) <= 1e-6


def _to_float(value: object) -> float | None:
    text = _value({"value": value}, "value").replace(",", ".")
    if not text:
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _format_number(value: float | None) -> str:
    if value is None:
        return ""
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _value(row: Mapping[str, object], key: str) -> str:
    return str(row.get(key, "") or "").strip()
