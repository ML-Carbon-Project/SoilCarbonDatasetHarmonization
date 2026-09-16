from __future__ import annotations

import csv
import hashlib
import io
import math
import unicodedata
import zipfile
from dataclasses import dataclass
from datetime import date
from io import BytesIO
from pathlib import Path

from carbono_solo.collector.geo import is_inside_brazil_bbox, normalize_year
from carbono_solo.collector.io import fetch_bytes
from carbono_solo.collector.lab_methods import hybras_assignment
from carbono_solo.collector.models import (
    CATALOG_ONLY,
    FAILED,
    IMPLEMENTED,
    InventoryRecord,
    SourceResult,
    TargetRecord,
)
from carbono_solo.collector.sources.soildata_tabular import _xlsx_sheet_texts


SOURCE_ID = "hybras_sgb"
SOURCE_NAME = "HYBRAS - SGB"
DATASET_ID = "hybras_v1_2020"
DATASET_URL = "https://www.sgb.gov.br/hybras"
DOWNLOAD_URL = (
    "https://rigeo.sgb.gov.br/bitstreams/"
    "080837a8-4ff5-414e-a0c6-e12179f04e32/download"
)
ARCHIVE_SHA256 = "E9164C2F74E9AD75033B99248B0EEE7BA426230150A499A6E6B3A285EEBCF618"
COMPLETE_WORKBOOK = "HYBRAS_COMPLETE_V1.xlsx"
SHAPE_WORKBOOK = "HYBRAS_excel_arquivoShape_modif18_7_2019.xlsx"
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ARCHIVE_PATH = PROJECT_ROOT / "data" / "raw" / "hybras" / "hybras_v1_2020.zip"
VAN_BEMMELEN_FACTOR = 1.724


STATE_ABBREVIATIONS = {
    "acre": "AC",
    "alagoas": "AL",
    "amapa": "AP",
    "amazonas": "AM",
    "bahia": "BA",
    "ceara": "CE",
    "distrito federal": "DF",
    "espirito santo": "ES",
    "goias": "GO",
    "maranhao": "MA",
    "mato grosso": "MT",
    "mato grosso do sul": "MS",
    "minas gerais": "MG",
    "para": "PA",
    "paraiba": "PB",
    "parana": "PR",
    "pernambuco": "PE",
    "piaui": "PI",
    "rio de janeiro": "RJ",
    "rio grande do norte": "RN",
    "rio grande do sul": "RS",
    "rondonia": "RO",
    "roraima": "RR",
    "santa catarina": "SC",
    "sao paulo": "SP",
    "sergipe": "SE",
    "tocantins": "TO",
}


@dataclass(frozen=True)
class HybrasParseResult:
    targets: list[TargetRecord]
    total_samples: int
    excluded_centroid_coordinates: int
    organic_matter_fallbacks: int


def collect_hybras(
    *,
    retrieved_at: str,
    use_network: bool = True,
    archive_path: Path = DEFAULT_ARCHIVE_PATH,
) -> SourceResult:
    try:
        if archive_path.exists():
            archive_bytes = archive_path.read_bytes()
        elif use_network:
            archive_bytes = fetch_bytes(DOWNLOAD_URL, timeout=240)
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            archive_path.write_bytes(archive_bytes)
        else:
            return SourceResult(
                inventory=[
                    _inventory_record(
                        retrieved_at=retrieved_at,
                        status=CATALOG_ONLY,
                        notes=f"Local archive not found: {archive_path}",
                    )
                ],
                targets=[],
            )

        actual_hash = hashlib.sha256(archive_bytes).hexdigest().upper()
        if actual_hash != ARCHIVE_SHA256:
            raise ValueError(
                "HYBRAS archive SHA256 mismatch: "
                f"expected {ARCHIVE_SHA256}, got {actual_hash}"
            )

        parsed = targets_from_archive(archive_bytes)
        notes = (
            f"Parsed {len(parsed.targets)} georeferenced carbon horizons with measured "
            "bulk density; "
            f"excluded {parsed.excluded_centroid_coordinates} target-capable horizons "
            "whose coordinates are municipality/state centroids; "
            f"{parsed.organic_matter_fallbacks} carbon values were derived from organic "
            f"matter using factor {VAN_BEMMELEN_FACTOR}. Raw samples: {parsed.total_samples}."
        )
        return SourceResult(
            inventory=[
                _inventory_record(
                    retrieved_at=retrieved_at,
                    status=IMPLEMENTED if parsed.targets else CATALOG_ONLY,
                    notes=notes,
                )
            ],
            targets=parsed.targets,
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


def targets_from_archive(archive_bytes: bytes) -> HybrasParseResult:
    with zipfile.ZipFile(BytesIO(archive_bytes)) as archive:
        complete_bytes = archive.read(COMPLETE_WORKBOOK)
        shape_bytes = archive.read(SHAPE_WORKBOOK)

    complete_sheets = _xlsx_sheet_texts(complete_bytes)
    shape_sheets = _xlsx_sheet_texts(shape_bytes)
    return targets_from_tables(
        complete_sheets["HYBRAS"],
        shape_sheets["ALL"],
    )


def targets_from_tables(
    complete_text: str,
    shape_text: str,
) -> HybrasParseResult:
    complete_rows = _complete_rows(complete_text)
    shape_rows = {
        _text(row.get("code")): row
        for row in csv.DictReader(io.StringIO(shape_text), delimiter="\t")
        if _text(row.get("code"))
    }
    targets: list[TargetRecord] = []
    excluded_centroids = 0
    organic_matter_fallbacks = 0

    for row in complete_rows:
        code = _text(row.get("code"))
        shape_row = shape_rows.get(code)
        if not code or shape_row is None:
            continue

        carbon = _carbon_g_kg(row)
        density = _number(row.get("bulk_den (g/cm3)"))
        depth_top = _number(row.get("top_depth"))
        depth_bottom = _number(row.get("bot_depth"))
        if (
            carbon is None
            or density is None
            or depth_top is None
            or depth_bottom is None
            or density <= 0
            or depth_bottom <= depth_top
        ):
            continue

        coordinate_comment = _text(shape_row.get("Comments_coordinates"))
        if "baricentrico" in _normalize(coordinate_comment):
            excluded_centroids += 1
            continue

        latitude = _number(shape_row.get("LatitudeOR"))
        longitude = _number(shape_row.get("LongitudeOR"))
        if latitude is None or longitude is None:
            continue
        if not is_inside_brazil_bbox(latitude=latitude, longitude=longitude):
            continue

        carbon_g_kg, carbon_source = carbon
        if carbon_source == "organic_matter":
            organic_matter_fallbacks += 1
        sample_date, sample_year = _sample_period(row)
        depth_top_text = _format_number(depth_top)
        depth_bottom_text = _format_number(depth_bottom)
        carbon_text = _format_number(carbon_g_kg)
        observation_id = f"HYBRAS:{code}"
        coordinate_quality = (
            "coordinate_utm_converted"
            if "utm" in coordinate_comment.lower()
            else "coordinate_from_source"
        )
        carbon_quality = (
            "organic_carbon_reported"
            if carbon_source == "organic_carbon"
            else "organic_matter_converted_to_carbon"
        )
        derivation_method = (
            "converted_hybras_organic_carbon_percent_to_g_kg"
            if carbon_source == "organic_carbon"
            else "derived_from_hybras_organic_matter_percent_van_bemmelen_1_724"
        )

        targets.append(
            TargetRecord(
                unified_observation_id=(
                    f"{SOURCE_ID}:{DATASET_ID}:{observation_id}:"
                    f"{depth_top_text}-{depth_bottom_text}:carbon_content"
                ),
                source_id=SOURCE_ID,
                dataset_id=DATASET_ID,
                source_observation_id=observation_id,
                latitude=latitude,
                longitude=longitude,
                coord_x=longitude,
                coord_y=latitude,
                crs="EPSG:4326",
                country="Brazil",
                state=_state_abbreviation(row.get("State")),
                sample_date=sample_date,
                sample_year=sample_year,
                sample_period_start=sample_year,
                sample_period_end=sample_year,
                depth_top_cm=depth_top_text,
                depth_bottom_cm=depth_bottom_text,
                layer_thickness_cm=_format_number(depth_bottom - depth_top),
                carbon_content_g_kg=carbon_text,
                carbon_stock_kg_m2="",
                carbon_stock_mg_ha="",
                bulk_density_g_cm3=_format_number(density),
                coarse_fragments_fraction="",
                carbon_stock_00_05="",
                carbon_stock_05_15="",
                carbon_stock_15_30="",
                carbon_stock_30_60="",
                carbon_stock_60_100="",
                target_kind="carbon_content",
                target_value=carbon_text,
                target_unit="g/kg",
                derivation_method=derivation_method,
                quality_flag=f"{coordinate_quality}|{carbon_quality}",
                duplicate_group_id="",
                duplicate_status="active",
                duplicate_resolution="unique",
                carbon_analysis_method=hybras_assignment(
                    row.get("OC_OM_Method")
                ).method,
            )
        )

    return HybrasParseResult(
        targets=targets,
        total_samples=len(complete_rows),
        excluded_centroid_coordinates=excluded_centroids,
        organic_matter_fallbacks=organic_matter_fallbacks,
    )


def _complete_rows(text: str) -> list[dict[str, str]]:
    rows = list(csv.reader(io.StringIO(text), delimiter="\t"))
    header_index = next(
        index
        for index, row in enumerate(rows)
        if "org_carb (%)" in row and "bulk_den (g/cm3)" in row
    )
    header = list(rows[header_index])
    header[0] = "code"
    return [dict(zip(header, row)) for row in rows[header_index + 1 :] if row]


def _carbon_g_kg(row: dict[str, str]) -> tuple[float, str] | None:
    organic_carbon_pct = _number(row.get("org_carb (%)"))
    if organic_carbon_pct is not None:
        return organic_carbon_pct * 10, "organic_carbon"
    organic_matter_pct = _number(row.get("org_mat (%)"))
    if organic_matter_pct is not None:
        return organic_matter_pct * 10 / VAN_BEMMELEN_FACTOR, "organic_matter"
    return None


def _sample_period(row: dict[str, str]) -> tuple[str, str]:
    year = normalize_year(row.get("year"))
    if not year:
        return "", ""
    month = _integer(row.get("month"))
    day = _integer(row.get("day"))
    if month is None or day is None:
        return "", year
    try:
        return date(int(year), month, day).isoformat(), year
    except ValueError:
        return "", year


def _inventory_record(*, retrieved_at: str, status: str, notes: str) -> InventoryRecord:
    return InventoryRecord(
        source_id=SOURCE_ID,
        source_name=SOURCE_NAME,
        dataset_id=DATASET_ID,
        title="Hydrophysical Database for Brazilian Soils (HYBRAS) version 1.0",
        url=DATASET_URL,
        country_scope="Brazil",
        access_type="official_sgb_zip_xlsx",
        brazil_filter="native_brazil_database_reliable_sample_coordinates",
        candidate_target_fields=(
            "organic carbon, organic matter, bulk density, layer depth, coordinates, "
            "texture, particle density, porosity, water retention, saturated conductivity"
        ),
        target_fit_score="high",
        implementation_status=status,
        license="not_specified_in_package",
        notes=notes,
        retrieved_at=retrieved_at,
    )


def _state_abbreviation(value: object) -> str:
    text = _text(value)
    if len(text) == 2:
        return text.upper()
    return STATE_ABBREVIATIONS.get(_normalize(text), "")


def _normalize(value: object) -> str:
    text = unicodedata.normalize("NFKD", _text(value))
    return "".join(char for char in text if not unicodedata.combining(char)).lower()


def _number(value: object) -> float | None:
    try:
        numeric = float(str(value or "").strip().replace(",", "."))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or numeric <= -900:
        return None
    return numeric


def _integer(value: object) -> int | None:
    numeric = _number(value)
    if numeric is None or not numeric.is_integer():
        return None
    return int(numeric)


def _format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _text(value: object) -> str:
    return str(value or "").strip()
