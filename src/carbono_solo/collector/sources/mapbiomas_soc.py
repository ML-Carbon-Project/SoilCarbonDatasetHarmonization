from __future__ import annotations

import csv
import math
from io import StringIO
from typing import Any

from carbono_solo.collector.geo import is_inside_brazil_bbox, normalize_year
from carbono_solo.collector.io import fetch_json, fetch_text
from carbono_solo.collector.lab_methods import mapbiomas_assignment
from carbono_solo.collector.models import (
    FAILED,
    IMPLEMENTED,
    InventoryRecord,
    SourceResult,
    TargetRecord,
)


DATASET_PID = "doi:10.60502/SoilData/IUZOAK"
DATASET_URL = "https://doi.org/10.60502/SoilData/IUZOAK"
DATAVERSE_API_URL = (
    "https://repositorio.soildata.mapbiomas.org/api/datasets/:persistentId/"
    "?persistentId=doi:10.60502/SoilData/IUZOAK"
)
DATAFILE_URL_TEMPLATE = "https://repositorio.soildata.mapbiomas.org/api/access/datafile/{file_id}"

SOURCE_ID = "mapbiomas_soc"
SOURCE_NAME = "MapBiomas Soil Organic Carbon Stock Training Data"
DEFAULT_TITLE = (
    "Training Field Soil Data for Annual Mapping of Soil Organic Carbon Stock "
    "(0-30 cm) in Brazil, 1985-2024 "
    "(MapBiomas Soil Collection 3, Beta Version)"
)
CANDIDATE_FIELDS = (
    "id, coord_x, coord_y, data_ano, profund_sup, profund_inf, carbono, "
    "dsi, vrc, soc_stock_gm2"
)


def _latest_version(metadata: dict[str, Any]) -> dict[str, Any]:
    latest = metadata.get("data", {}).get("latestVersion", {})
    return latest if isinstance(latest, dict) else {}


def _citation_title(metadata: dict[str, Any]) -> str:
    fields = (
        _latest_version(metadata)
        .get("metadataBlocks", {})
        .get("citation", {})
        .get("fields", [])
    )
    if not isinstance(fields, list):
        return DEFAULT_TITLE
    for field in fields:
        if isinstance(field, dict) and field.get("typeName") == "title":
            title = str(field.get("value", "")).strip()
            if title:
                return title
    return DEFAULT_TITLE


def _stock_file_id(metadata: dict[str, Any]) -> int:
    files = _latest_version(metadata).get("files", [])
    if not isinstance(files, list):
        files = []
    for file_info in files:
        file_id = _file_id_if_matches(file_info, require_modeling=True)
        if file_id is not None:
            return file_id
    for file_info in files:
        file_id = _file_id_if_matches(file_info, require_modeling=False)
        if file_id is not None:
            return file_id
    raise ValueError("MapBiomas SOC stock file was not found in Dataverse metadata.")


def _file_id_if_matches(file_info: object, require_modeling: bool) -> int | None:
    if not isinstance(file_info, dict):
        return None

    label = str(file_info.get("label", "")).lower()
    description = str(file_info.get("description", "")).lower()
    filename = ""
    data_file = file_info.get("dataFile", {})
    if isinstance(data_file, dict):
        filename = str(data_file.get("filename", "")).lower()

    search_text = " ".join([label, description, filename])
    is_modeling_file = (
        "c3_soildata_soc_modeling" in search_text
        or "spatio-temporal modeling" in search_text
    )
    is_stock_file = (
        "soil-organic-carbon-stock" in search_text
        or "organic carbon stock" in search_text
        or is_modeling_file
    )
    if require_modeling and not is_modeling_file:
        return None
    if not require_modeling and not is_stock_file:
        return None
    if not isinstance(data_file, dict):
        return None
    try:
        return int(data_file["id"])
    except (KeyError, TypeError, ValueError):
        return None


def dataverse_inventory_from_metadata(
    metadata: dict[str, Any],
    retrieved_at: str,
) -> tuple[InventoryRecord, int]:
    latest = _latest_version(metadata)
    license_info = latest.get("license", {})
    license_name = ""
    if isinstance(license_info, dict):
        license_name = str(license_info.get("name", "")).strip()
    elif license_info:
        license_name = str(license_info).strip()

    file_id = _stock_file_id(metadata)
    inventory = InventoryRecord(
        source_id=SOURCE_ID,
        source_name=SOURCE_NAME,
        dataset_id=DATASET_PID,
        title=_citation_title(metadata),
        url=DATASET_URL,
        country_scope="Brazil",
        access_type="dataverse_api",
        brazil_filter="native_brazil_dataset",
        candidate_target_fields=CANDIDATE_FIELDS,
        target_fit_score="high",
        implementation_status=IMPLEMENTED,
        license=license_name,
        notes=(
            f"Dataverse file id {file_id}; Collection 3 has 16,013 georeferenced "
            "points and 31,047 soil layers; target is layer SOC stock in g/m2."
        ),
        retrieved_at=retrieved_at,
    )
    return inventory, file_id


def _format_float(value: float) -> str:
    return str(round(value, 6)).rstrip("0").rstrip(".")


def _format_stock_value(value: float) -> str:
    text = _format_float(value)
    if "." in text:
        return text
    return f"{text}.0"


def parse_mapbiomas_tsv(text: str) -> list[TargetRecord]:
    reader = csv.DictReader(StringIO(text), delimiter="\t")
    records: list[TargetRecord] = []
    for row in reader:
        latitude = _row_float(row, "coord_y", "latitude")
        longitude = _row_float(row, "coord_x", "longitude")
        stock_g_m2 = _row_float(row, "soc_stock_gm2", "soc_stock_g_m2")
        depth_top = _row_float(row, "profund_sup", "depth_top_cm")
        depth_bottom = _row_float(row, "profund_inf", "depth_bottom_cm")
        if (
            latitude is None
            or longitude is None
            or stock_g_m2 is None
            or depth_top is None
            or depth_bottom is None
        ):
            continue

        if not is_inside_brazil_bbox(latitude=latitude, longitude=longitude):
            continue

        point_id = str(row.get("id") or row.get("point_id") or "").strip()
        if not point_id:
            continue

        year = normalize_year(row.get("data_ano") or row.get("year") or "")
        if not year:
            continue

        layer_thickness = _row_float(row, "espessura", "layer_thickness_cm")
        if layer_thickness is None:
            layer_thickness = depth_bottom - depth_top

        stock_kg_m2 = stock_g_m2 / 1000.0
        stock_mg_ha = stock_g_m2 * 0.01
        depth_top_text = _format_float(depth_top)
        depth_bottom_text = _format_float(depth_bottom)
        unified_id = (
            f"{SOURCE_ID}:{point_id}:{year}:"
            f"{depth_top_text}-{depth_bottom_text}:soc_stock_layer"
        )

        records.append(
            TargetRecord(
                unified_observation_id=unified_id,
                source_id=SOURCE_ID,
                dataset_id=DATASET_PID,
                source_observation_id=point_id,
                latitude=latitude,
                longitude=longitude,
                coord_x=longitude,
                coord_y=latitude,
                crs="EPSG:4326",
                country="Brazil",
                state="",
                sample_date="",
                sample_year=year,
                sample_period_start=year,
                sample_period_end=year,
                depth_top_cm=depth_top_text,
                depth_bottom_cm=depth_bottom_text,
                layer_thickness_cm=_format_float(layer_thickness),
                carbon_content_g_kg=_row_text(row, "carbono", "carbon_content_g_kg"),
                carbon_stock_kg_m2=_format_stock_value(stock_kg_m2),
                carbon_stock_mg_ha=_format_stock_value(stock_mg_ha),
                bulk_density_g_cm3=_row_text(row, "dsi", "bulk_density_g_cm3"),
                coarse_fragments_fraction=_row_text(row, "vrc", "coarse_fragments_fraction"),
                carbon_stock_00_05="",
                carbon_stock_05_15="",
                carbon_stock_15_30="",
                carbon_stock_30_60="",
                carbon_stock_60_100="",
                target_kind="soc_stock_layer",
                target_value=_format_float(stock_g_m2),
                target_unit="g/m2",
                derivation_method="reported_by_source",
                quality_flag="ok",
                duplicate_group_id="",
                duplicate_status="active",
                duplicate_resolution="unique",
                carbon_analysis_method=mapbiomas_assignment(point_id).method,
            )
        )
    return records


def _row_float(row: dict[str, str], *keys: str) -> float | None:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip().replace(",", ".")
        if not text:
            continue
        try:
            number = float(text)
        except ValueError:
            continue
        if math.isfinite(number):
            return number
    return None


def _row_text(row: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip().replace(",", ".")
        if text:
            return text
    return ""


def _failed_inventory(retrieved_at: str, license_name: str, notes: str) -> InventoryRecord:
    return InventoryRecord(
        source_id=SOURCE_ID,
        source_name=SOURCE_NAME,
        dataset_id=DATASET_PID,
        title=DEFAULT_TITLE,
        url=DATASET_URL,
        country_scope="Brazil",
        access_type="dataverse_api",
        brazil_filter="native_brazil_dataset",
        candidate_target_fields=CANDIDATE_FIELDS,
        target_fit_score="high",
        implementation_status=FAILED,
        license=license_name,
        notes=notes,
        retrieved_at=retrieved_at,
    )


def collect_mapbiomas_soc(retrieved_at: str, use_network: bool = True) -> SourceResult:
    if not use_network:
        inventory = _failed_inventory(
            retrieved_at=retrieved_at,
            license_name="CC BY 4.0",
            notes=(
                "Offline mode: target download skipped; run without --offline "
                "to download targets."
            ),
        )
        return SourceResult(inventory=[inventory], targets=[])

    try:
        metadata = fetch_json(DATAVERSE_API_URL)
        if not isinstance(metadata, dict):
            raise ValueError("Dataverse metadata response was not an object.")
        inventory, file_id = dataverse_inventory_from_metadata(metadata, retrieved_at)
        target_text = fetch_text(DATAFILE_URL_TEMPLATE.format(file_id=file_id))
        targets = parse_mapbiomas_tsv(target_text)
        return SourceResult(inventory=[inventory], targets=targets)
    except Exception as exc:
        inventory = _failed_inventory(
            retrieved_at=retrieved_at,
            license_name="",
            notes=str(exc),
        )
        return SourceResult(inventory=[inventory], targets=[])
