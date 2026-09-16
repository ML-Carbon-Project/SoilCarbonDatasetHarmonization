from __future__ import annotations

import csv
import io
import math
import zipfile
from io import BytesIO

from carbono_solo.collector.geo import is_inside_brazil_bbox, normalize_year
from carbono_solo.collector.io import fetch_bytes
from carbono_solo.collector.lab_methods import israd_assignment
from carbono_solo.collector.models import (
    CATALOG_ONLY,
    FAILED,
    IMPLEMENTED,
    InventoryRecord,
    SourceResult,
    TargetRecord,
)


SOURCE_ID = "israd_brazil"
SOURCE_NAME = "ISRaD Brazil subset"
DATASET_ID = "israd_flat_layer"
DOWNLOAD_URL = (
    "https://github.com/International-Soil-Radiocarbon-Database/ISRaD/raw/main/"
    "ISRaD_data_files/database/ISRaD_database_files.zip"
)


def collect_israd_brazil(
    *,
    retrieved_at: str,
    use_network: bool = True,
) -> SourceResult:
    if not use_network:
        return SourceResult(inventory=[], targets=[])

    try:
        archive_bytes = fetch_bytes(DOWNLOAD_URL, timeout=180)
        targets = _targets_from_zip(archive_bytes)
        return SourceResult(
            inventory=[
                _inventory_record(
                    retrieved_at=retrieved_at,
                    status=IMPLEMENTED if targets else CATALOG_ONLY,
                    notes=(
                        f"Parsed {len(targets)} target rows from ISRaD "
                        "flat layer CSV, filtered to Brazil bbox."
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


def targets_from_layer_csv(text: str, *, file_name: str) -> list[TargetRecord]:
    reader = csv.DictReader(io.StringIO(text))
    records: list[TargetRecord] = []
    for row in reader:
        latitude = _first_float(row, ["pro_lat", "site_lat"])
        longitude = _first_float(row, ["pro_long", "site_long"])
        if latitude is None or longitude is None:
            continue
        if not is_inside_brazil_bbox(latitude=latitude, longitude=longitude):
            continue

        depth_top = _first_float(row, ["lyr_top"])
        depth_bottom = _first_float(row, ["lyr_bot"])
        measurement = _carbon_measurement(row)
        if depth_top is None or depth_bottom is None or measurement is None:
            continue

        entry_name = _text(row.get("entry_name")) or "israd"
        profile_name = _text(row.get("pro_name")) or _text(row.get("site_name"))
        layer_name = _text(row.get("lyr_name")) or f"{depth_top}-{depth_bottom}"
        observation_id = f"{entry_name}:{profile_name}:{layer_name}"
        sample_year = normalize_year(_text(row.get("lyr_obs_date_y")))
        carbon_g_kg, target_kind, derivation_method = measurement
        density = _first_float(row, ["lyr_bd_samp", "lyr_bd_tot"])
        coarse = _coarse_fraction(_first_float(row, ["lyr_coarse_tot"]))
        depth_top_text = _format_number(depth_top)
        depth_bottom_text = _format_number(depth_bottom)
        carbon_text = _format_number(carbon_g_kg)
        unified_id = (
            f"{SOURCE_ID}:{DATASET_ID}:{file_name}:{observation_id}:"
            f"{depth_top_text}-{depth_bottom_text}:{target_kind}"
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
                sample_date="",
                sample_year=sample_year,
                sample_period_start=sample_year,
                sample_period_end=sample_year,
                depth_top_cm=depth_top_text,
                depth_bottom_cm=depth_bottom_text,
                layer_thickness_cm=_format_number(depth_bottom - depth_top),
                carbon_content_g_kg=carbon_text,
                carbon_stock_kg_m2="",
                carbon_stock_mg_ha="",
                bulk_density_g_cm3="" if density is None else _format_number(density),
                coarse_fragments_fraction=(
                    "" if coarse is None else _format_number(coarse)
                ),
                carbon_stock_00_05="",
                carbon_stock_05_15="",
                carbon_stock_15_30="",
                carbon_stock_30_60="",
                carbon_stock_60_100="",
                target_kind=target_kind,
                target_value=carbon_text,
                target_unit="g/kg",
                derivation_method=derivation_method,
                quality_flag="ok",
                duplicate_group_id="",
                duplicate_status="active",
                duplicate_resolution="unique",
                carbon_analysis_method=israd_assignment(
                    entry_name,
                    sample_year,
                ).method,
            )
        )
    return records


def _targets_from_zip(archive_bytes: bytes) -> list[TargetRecord]:
    try:
        with zipfile.ZipFile(BytesIO(archive_bytes)) as archive:
            layer_file = _layer_csv_name(archive.namelist())
            if layer_file is None:
                raise ValueError("ISRaD flat layer CSV not found in archive.")
            text = archive.read(layer_file).decode("utf-8-sig")
    except zipfile.BadZipFile as exc:
        raise ValueError("Invalid ISRaD ZIP archive.") from exc
    return targets_from_layer_csv(text, file_name=layer_file)


def _layer_csv_name(names: list[str]) -> str | None:
    for name in names:
        lowered = name.lower()
        if "data_flat_layer" in lowered and lowered.endswith(".csv"):
            return name
    return None


def _carbon_measurement(row: dict[str, str]) -> tuple[float, str, str] | None:
    organic = _to_float(row.get("lyr_c_org"))
    if organic is not None:
        return organic * 10.0, "carbon_content", "reported_israd_lyr_c_org_percent_to_g_kg"
    total = _to_float(row.get("lyr_c_tot"))
    if total is not None:
        return (
            total * 10.0,
            "total_carbon_content",
            "reported_israd_lyr_c_tot_percent_to_g_kg",
        )
    return None


def _inventory_record(*, retrieved_at: str, status: str, notes: str) -> InventoryRecord:
    return InventoryRecord(
        source_id=SOURCE_ID,
        source_name=SOURCE_NAME,
        dataset_id=DATASET_ID,
        title="ISRaD flat layer records filtered to Brazil",
        url=DOWNLOAD_URL,
        country_scope="Brazil",
        access_type="github_zip_csv_download",
        brazil_filter="coordinate_bbox_filter",
        candidate_target_fields=(
            "layer organic carbon, total carbon, bulk density, coarse fragments, "
            "layer depth, profile coordinates"
        ),
        target_fit_score="medium",
        implementation_status=status,
        license="CC BY 4.0",
        notes=notes,
        retrieved_at=retrieved_at,
    )


def _first_float(row: dict[str, str], keys: list[str]) -> float | None:
    for key in keys:
        value = _to_float(row.get(key))
        if value is not None:
            return value
    return None


def _to_float(value: object) -> float | None:
    text = _text(value)
    if not text or text.lower() in {"na", "nan", "null", "none"}:
        return None
    try:
        numeric = float(text)
    except ValueError:
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _coarse_fraction(value: float | None) -> float | None:
    if value is None:
        return None
    if value > 1:
        return value / 100.0
    return value


def _format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _text(value: object) -> str:
    return str(value or "").strip()
