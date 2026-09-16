from __future__ import annotations

import csv
import io
import math
import re
import unicodedata
import zipfile
from dataclasses import dataclass
from typing import Iterable
from xml.etree import ElementTree

from carbono_solo.collector.geo import is_inside_brazil_bbox, normalize_year
from carbono_solo.collector.lab_methods import soildata_assignment
from carbono_solo.collector.models import TargetRecord


SOURCE_ID = "soildata_dataset"
SOURCE_NAME = "SoilData Dataverse Dataset"

NA_VALUES = {"", "na", "nan", "null", "none", "nd"}


@dataclass(frozen=True)
class ParsedCarbon:
    value_g_kg: float
    source_column: str


@dataclass(frozen=True)
class ParsedTargetValue:
    target_kind: str
    target_value: float
    target_unit: str
    carbon_content_g_kg: float | None = None
    carbon_stock_mg_ha: float | None = None
    derivation_method: str = "reported_or_unit_converted_by_source"
    quality_flag: str = "ok"


def parse_tabular_targets(
    text: str,
    *,
    dataset_id: str,
    file_id: str,
    file_name: str,
    retrieved_at: str,
) -> list[TargetRecord]:
    del retrieved_at
    rows = _read_rows(text)
    records: list[TargetRecord] = []
    last_latitude: float | None = None
    last_longitude: float | None = None

    for row_index, row in enumerate(rows, start=1):
        latitude, longitude = _coordinates(row)
        if latitude is not None and longitude is not None:
            last_latitude, last_longitude = latitude, longitude
        else:
            latitude, longitude = last_latitude, last_longitude

        if latitude is None or longitude is None:
            continue
        if not is_inside_brazil_bbox(latitude=latitude, longitude=longitude):
            continue

        depth_top, depth_bottom = _depth(row)
        if depth_top == "" or depth_bottom == "":
            continue

        measurements = _target_measurements(row)
        if not measurements:
            continue

        observation_id = _first_text(
            row,
            [
                "observacao_id",
                "identificacao do evento",
                "codigo",
                "codigo",
                "cod",
            ],
        ) or f"{file_id}:{row_index}"
        layer_id = _first_text(
            row,
            [
                "identificacao da camada",
                "camada_id",
                "prof",
                "profundidade",
            ],
        ) or f"{depth_top}-{depth_bottom}"

        density = _bulk_density(row)
        coarse = _coarse_fraction(row)
        sample_date, sample_year, period_start, period_end = _sample_period(row)

        for measurement in measurements:
            records.append(
                _target_record(
                    dataset_id=dataset_id,
                    file_ref=f"{file_id}:{file_name}",
                    observation_id=observation_id,
                    layer_id=layer_id,
                    latitude=latitude,
                    longitude=longitude,
                    state=_first_text(row, ["estado", "estado uf", "estado_id"]),
                    sample_date=sample_date,
                    sample_year=sample_year,
                    sample_period_start=period_start,
                    sample_period_end=period_end,
                    depth_top_cm=depth_top,
                    depth_bottom_cm=depth_bottom,
                    carbon_content_g_kg=measurement.carbon_content_g_kg,
                    carbon_stock_mg_ha=measurement.carbon_stock_mg_ha,
                    target_kind=measurement.target_kind,
                    target_value=measurement.target_value,
                    target_unit=measurement.target_unit,
                    derivation_method=measurement.derivation_method,
                    quality_flag=measurement.quality_flag,
                    bulk_density_g_cm3=density,
                    coarse_fragments_fraction=coarse,
                )
            )

    return records


def parse_xlsx_targets(
    data: bytes,
    *,
    dataset_id: str,
    file_id: str,
    file_name: str,
    retrieved_at: str,
) -> list[TargetRecord]:
    sheets = _xlsx_sheet_texts(data)
    event_sheet = _sheet_by_name(sheets, "evento")
    layer_sheet = _sheet_by_name(sheets, "camada")
    targets: list[TargetRecord] = []

    if event_sheet and layer_sheet:
        targets.extend(
            parse_paired_event_layer_tables(
                event_sheet[1],
                layer_sheet[1],
                dataset_id=dataset_id,
                event_file_id=f"{file_id}:{event_sheet[0]}",
                layer_file_id=f"{file_id}:{layer_sheet[0]}",
                retrieved_at=retrieved_at,
            )
        )

    used_sheets = {event_sheet[0], layer_sheet[0]} if event_sheet and layer_sheet else set()
    for sheet_name, sheet_text in sheets.items():
        if sheet_name in used_sheets:
            continue
        if _is_auxiliary_sheet(sheet_name):
            continue
        targets.extend(
            parse_wide_layer_targets(
                sheet_text,
                dataset_id=dataset_id,
                file_id=f"{file_id}:{sheet_name}",
                file_name=file_name,
                retrieved_at=retrieved_at,
            )
        )
        targets.extend(
            parse_tabular_targets(
                sheet_text,
                dataset_id=dataset_id,
                file_id=f"{file_id}:{sheet_name}",
                file_name=file_name,
                retrieved_at=retrieved_at,
            )
        )

    return targets


def parse_wide_layer_targets(
    text: str,
    *,
    dataset_id: str,
    file_id: str,
    file_name: str,
    retrieved_at: str,
) -> list[TargetRecord]:
    del retrieved_at
    matrix = _read_matrix(text)
    if len(matrix) < 3:
        return []

    header_index = _wide_header_index(matrix)
    if header_index is None or header_index == 0:
        return []

    context_row = matrix[header_index - 1]
    header = [_normalize_key(value) for value in matrix[header_index]]
    try:
        id_index = header.index("id")
        x_index = header.index("x")
        y_index = header.index("y")
    except ValueError:
        return []

    depth_occurrences: dict[str, list[int]] = {}
    for column_index, label in enumerate(matrix[header_index]):
        depth = _depth_from_text(label)
        if depth is not None:
            depth_occurrences.setdefault(depth, []).append(column_index)

    zone, south = _utm_zone_from_context(context_row)
    if zone is None:
        return []

    records: list[TargetRecord] = []
    for row_index, row in enumerate(matrix[header_index + 1 :], start=1):
        if len(row) <= max(id_index, x_index, y_index):
            continue
        observation_id = _clean_text(row[id_index])
        x = _to_optional_float(row[x_index])
        y = _to_optional_float(row[y_index])
        if not observation_id or x is None or y is None:
            continue
        latitude, longitude = _utm_to_latlon(x, y, zone=zone, south=south)
        if not is_inside_brazil_bbox(latitude=latitude, longitude=longitude):
            continue

        for depth_text, columns in depth_occurrences.items():
            if len(columns) < 2:
                continue
            density_column = columns[0]
            carbon_column = columns[1]
            if carbon_column >= len(row):
                continue
            carbon = _to_optional_float(row[carbon_column])
            if carbon is None:
                continue
            density = (
                _to_optional_float(row[density_column])
                if density_column < len(row)
                else None
            )
            depth_top, depth_bottom = depth_text.split("-", maxsplit=1)
            records.append(
                _target_record(
                    dataset_id=dataset_id,
                    file_ref=f"{file_id}:{file_name}",
                    observation_id=observation_id,
                    layer_id=f"{depth_text}:{row_index}",
                    latitude=latitude,
                    longitude=longitude,
                    state="",
                    sample_date="",
                    sample_year="",
                    sample_period_start="",
                    sample_period_end="",
                    depth_top_cm=depth_top,
                    depth_bottom_cm=depth_bottom,
                    carbon_content_g_kg=carbon,
                    bulk_density_g_cm3=density,
                    coarse_fragments_fraction=None,
                )
            )

    return records


def parse_paired_event_layer_tables(
    event_text: str,
    layer_text: str,
    *,
    dataset_id: str,
    event_file_id: str,
    layer_file_id: str,
    retrieved_at: str,
) -> list[TargetRecord]:
    del retrieved_at
    events = {}
    for event_row in _read_rows(event_text):
        event_keys = _event_keys(event_row)
        latitude, longitude = _coordinates(event_row)
        if not event_keys or latitude is None or longitude is None:
            continue
        if not is_inside_brazil_bbox(latitude=latitude, longitude=longitude):
            continue
        for event_key in event_keys:
            events[event_key] = event_row

    records: list[TargetRecord] = []
    for row_index, layer_row in enumerate(_read_rows(layer_text), start=1):
        event_id = _event_id(layer_row)
        event_row = events.get(event_id)
        if not event_row:
            continue

        depth_top, depth_bottom = _depth(layer_row)
        if depth_top == "" or depth_bottom == "":
            continue

        measurements = _target_measurements(layer_row)
        if not measurements:
            continue

        latitude, longitude = _coordinates(event_row)
        if latitude is None or longitude is None:
            continue

        sample_date, sample_year, period_start, period_end = _sample_period(event_row)
        layer_id = _first_text(
            layer_row,
            ["identificacao da camada", "camada_id"],
        ) or f"{layer_file_id}:{row_index}"
        for measurement in measurements:
            records.append(
                _target_record(
                    dataset_id=dataset_id,
                    file_ref=f"{event_file_id}+{layer_file_id}",
                    observation_id=event_id,
                    layer_id=layer_id,
                    latitude=latitude,
                    longitude=longitude,
                    state=_first_text(event_row, ["estado", "estado uf", "estado_id"]),
                    sample_date=sample_date,
                    sample_year=sample_year,
                    sample_period_start=period_start,
                    sample_period_end=period_end,
                    depth_top_cm=depth_top,
                    depth_bottom_cm=depth_bottom,
                    carbon_content_g_kg=measurement.carbon_content_g_kg,
                    carbon_stock_mg_ha=measurement.carbon_stock_mg_ha,
                    target_kind=measurement.target_kind,
                    target_value=measurement.target_value,
                    target_unit=measurement.target_unit,
                    derivation_method=measurement.derivation_method,
                    quality_flag=measurement.quality_flag,
                    bulk_density_g_cm3=_bulk_density(layer_row),
                    coarse_fragments_fraction=_coarse_fraction(layer_row),
                )
            )

    return records


def _target_record(
    *,
    dataset_id: str,
    file_ref: str,
    observation_id: str,
    layer_id: str,
    latitude: float,
    longitude: float,
    state: str,
    sample_date: str,
    sample_year: str,
    sample_period_start: str,
    sample_period_end: str,
    depth_top_cm: str,
    depth_bottom_cm: str,
    carbon_content_g_kg: float | None,
    bulk_density_g_cm3: float | None,
    coarse_fragments_fraction: float | None,
    carbon_stock_mg_ha: float | None = None,
    target_kind: str = "carbon_content",
    target_value: float | None = None,
    target_unit: str = "g/kg",
    derivation_method: str = "reported_or_unit_converted_by_source",
    quality_flag: str = "ok",
) -> TargetRecord:
    thickness = _format_number(_to_float(depth_bottom_cm) - _to_float(depth_top_cm))
    content = (
        ""
        if carbon_content_g_kg is None
        else _format_number(carbon_content_g_kg)
    )
    stock_mg_ha = (
        ""
        if carbon_stock_mg_ha is None
        else _format_number(carbon_stock_mg_ha)
    )
    if target_value is None:
        target_value = carbon_content_g_kg
    target_value_text = "" if target_value is None else _format_number(target_value)
    density = "" if bulk_density_g_cm3 is None else _format_number(bulk_density_g_cm3)
    coarse = (
        ""
        if coarse_fragments_fraction is None
        else _format_number(coarse_fragments_fraction)
    )
    unified_id = (
        f"{SOURCE_ID}:{dataset_id}:{file_ref}:{observation_id}:"
        f"{layer_id}:{depth_top_cm}-{depth_bottom_cm}:{target_kind}"
    )
    return TargetRecord(
        unified_observation_id=unified_id,
        source_id=SOURCE_ID,
        dataset_id=dataset_id,
        source_observation_id=observation_id,
        latitude=latitude,
        longitude=longitude,
        coord_x=longitude,
        coord_y=latitude,
        crs="EPSG:4326",
        country="Brazil",
        state=state,
        sample_date=sample_date,
        sample_year=sample_year,
        sample_period_start=sample_period_start,
        sample_period_end=sample_period_end,
        depth_top_cm=depth_top_cm,
        depth_bottom_cm=depth_bottom_cm,
        layer_thickness_cm=thickness,
        carbon_content_g_kg=content,
        carbon_stock_kg_m2="",
        carbon_stock_mg_ha=stock_mg_ha,
        bulk_density_g_cm3=density,
        coarse_fragments_fraction=coarse,
        carbon_stock_00_05="",
        carbon_stock_05_15="",
        carbon_stock_15_30="",
        carbon_stock_30_60="",
        carbon_stock_60_100="",
        target_kind=target_kind,
        target_value=target_value_text,
        target_unit=target_unit,
        derivation_method=derivation_method,
        quality_flag=quality_flag,
        duplicate_group_id="",
        duplicate_status="active",
        duplicate_resolution="unique",
        carbon_analysis_method=soildata_assignment(dataset_id).method,
    )


def _read_rows(text: str) -> list[dict[str, str]]:
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="\t;,")
    except csv.Error:
        dialect = csv.excel_tab if "\t" in sample else csv.excel
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    rows = []
    for raw_row in reader:
        normalized = {}
        for key, value in raw_row.items():
            if key is None:
                continue
            normalized[_normalize_key(key)] = _clean_text(value)
        rows.append(normalized)
    return rows


def _read_matrix(text: str) -> list[list[str]]:
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="\t;,")
    except csv.Error:
        dialect = csv.excel_tab if "\t" in sample else csv.excel
    return [
        [_clean_text(value) for value in row]
        for row in csv.reader(io.StringIO(text), dialect=dialect)
    ]


def _xlsx_sheet_texts(data: bytes) -> dict[str, str]:
    archive = zipfile.ZipFile(io.BytesIO(data))
    namespace = {
        "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    }
    shared_strings = _xlsx_shared_strings(archive, namespace)
    workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
    relationships = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    relationship_targets = {
        relationship.attrib["Id"]: relationship.attrib["Target"]
        for relationship in relationships
    }

    sheets: dict[str, str] = {}
    for sheet in workbook.findall("a:sheets/a:sheet", namespace):
        sheet_name = sheet.attrib.get("name", "").strip()
        relation_id = sheet.attrib.get(
            "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id",
            "",
        )
        target = relationship_targets.get(relation_id, "")
        sheet_path = _xlsx_target_path(target)
        if not sheet_name or sheet_path not in archive.namelist():
            continue
        sheets[sheet_name] = _xlsx_sheet_to_tsv(
            archive.read(sheet_path),
            shared_strings,
            namespace,
        )
    return sheets


def _xlsx_shared_strings(
    archive: zipfile.ZipFile,
    namespace: dict[str, str],
) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
    strings = []
    for item in root.findall("a:si", namespace):
        strings.append("".join(text.text or "" for text in item.findall(".//a:t", namespace)))
    return strings


def _xlsx_target_path(target: str) -> str:
    normalized = target.lstrip("/")
    if normalized.startswith("xl/"):
        return normalized
    return f"xl/{normalized}"


def _xlsx_sheet_to_tsv(
    sheet_xml: bytes,
    shared_strings: list[str],
    namespace: dict[str, str],
) -> str:
    root = ElementTree.fromstring(sheet_xml)
    lines = []
    for row in root.findall(".//a:sheetData/a:row", namespace):
        values_by_index: dict[int, str] = {}
        for cell in row.findall("a:c", namespace):
            cell_ref = cell.attrib.get("r", "")
            column_index = _column_index(cell_ref)
            if column_index < 1:
                continue
            values_by_index[column_index] = _xlsx_cell_value(
                cell,
                shared_strings,
                namespace,
            )
        if not values_by_index:
            lines.append("")
            continue
        width = max(values_by_index)
        lines.append("\t".join(values_by_index.get(index, "") for index in range(1, width + 1)))
    return "\n".join(lines)


def _xlsx_cell_value(
    cell: ElementTree.Element,
    shared_strings: list[str],
    namespace: dict[str, str],
) -> str:
    cell_type = cell.attrib.get("t")
    value = cell.find("a:v", namespace)
    if value is not None and value.text is not None:
        text = value.text
        if cell_type == "s":
            index = int(text)
            return shared_strings[index] if index < len(shared_strings) else text
        return text
    if cell_type == "inlineStr":
        return "".join(text.text or "" for text in cell.findall(".//a:t", namespace))
    return ""


def _column_index(cell_ref: str) -> int:
    letters = re.sub(r"[^A-Z]", "", cell_ref.upper())
    index = 0
    for letter in letters:
        index = index * 26 + ord(letter) - 64
    return index


def _wide_header_index(matrix: list[list[str]]) -> int | None:
    for index, row in enumerate(matrix):
        normalized = {_normalize_key(value) for value in row}
        if {"id", "x", "y"}.issubset(normalized):
            return index
    return None


def _utm_zone_from_context(row: list[str]) -> tuple[int | None, bool]:
    context = " ".join(row).lower()
    match = re.search(r"\b(?:z|zone|zona)\s*(\d{1,2})\s*([ns])?", context)
    if not match:
        return None, True
    zone = int(match.group(1))
    hemisphere = (match.group(2) or "s").lower()
    return zone, hemisphere != "n"


def _utm_to_latlon(
    easting: float,
    northing: float,
    *,
    zone: int,
    south: bool,
) -> tuple[float, float]:
    axis = 6378137.0
    eccentricity_squared = 0.00669438
    scale = 0.9996
    eccentricity_prime_squared = eccentricity_squared / (1.0 - eccentricity_squared)
    e1 = (
        (1.0 - math.sqrt(1.0 - eccentricity_squared))
        / (1.0 + math.sqrt(1.0 - eccentricity_squared))
    )

    x = easting - 500000.0
    y = northing - (10000000.0 if south else 0.0)
    longitude_origin = (zone - 1.0) * 6.0 - 180.0 + 3.0
    meridional_arc = y / scale
    mu = meridional_arc / (
        axis
        * (
            1.0
            - eccentricity_squared / 4.0
            - 3.0 * eccentricity_squared**2 / 64.0
            - 5.0 * eccentricity_squared**3 / 256.0
        )
    )

    phi1 = (
        mu
        + (3.0 * e1 / 2.0 - 27.0 * e1**3 / 32.0) * math.sin(2.0 * mu)
        + (21.0 * e1**2 / 16.0 - 55.0 * e1**4 / 32.0) * math.sin(4.0 * mu)
        + (151.0 * e1**3 / 96.0) * math.sin(6.0 * mu)
        + (1097.0 * e1**4 / 512.0) * math.sin(8.0 * mu)
    )
    n1 = axis / math.sqrt(1.0 - eccentricity_squared * math.sin(phi1) ** 2)
    t1 = math.tan(phi1) ** 2
    c1 = eccentricity_prime_squared * math.cos(phi1) ** 2
    r1 = (
        axis
        * (1.0 - eccentricity_squared)
        / (1.0 - eccentricity_squared * math.sin(phi1) ** 2) ** 1.5
    )
    d = x / (n1 * scale)

    latitude = phi1 - (
        n1
        * math.tan(phi1)
        / r1
        * (
            d**2 / 2.0
            - (
                5.0
                + 3.0 * t1
                + 10.0 * c1
                - 4.0 * c1**2
                - 9.0 * eccentricity_prime_squared
            )
            * d**4
            / 24.0
            + (
                61.0
                + 90.0 * t1
                + 298.0 * c1
                + 45.0 * t1**2
                - 252.0 * eccentricity_prime_squared
                - 3.0 * c1**2
            )
            * d**6
            / 720.0
        )
    )
    longitude = (
        d
        - (1.0 + 2.0 * t1 + c1) * d**3 / 6.0
        + (
            5.0
            - 2.0 * c1
            + 28.0 * t1
            - 3.0 * c1**2
            + 8.0 * eccentricity_prime_squared
            + 24.0 * t1**2
        )
        * d**5
        / 120.0
    ) / math.cos(phi1)

    return math.degrees(latitude), longitude_origin + math.degrees(longitude)


def _coordinates(row: dict[str, str]) -> tuple[float | None, float | None]:
    latitude = _first_coordinate(
        row,
        [
            "latitude grau",
            "latitude",
            "coord_y",
            "y m",
            "y",
            "coordenadasgeograficas",
        ],
    )
    longitude = _first_coordinate(
        row,
        ["longitude grau", "longitude", "coord_x", "x m", "x"],
    )
    if longitude is None and "coordenadasgeograficas" in row:
        longitude = _to_optional_float(row.get("d", ""))
    if latitude is None:
        latitude = _dms_coordinate(row, prefix="lat")
    if longitude is None:
        longitude = _dms_coordinate(row, prefix="long")
    return latitude, longitude


def _depth(row: dict[str, str]) -> tuple[str, str]:
    top = _first_float(
        row,
        [
            "profundidade superior cm",
            "profundidade inicial cm",
            "profundidde inicial cm",
            "profund_sup",
            "depth_top_cm",
        ],
    )
    bottom = _first_float(
        row,
        [
            "profundidade inferior cm",
            "profundidade final cm",
            "profundidde final cm",
            "profund_inf",
            "depth_bottom_cm",
        ],
    )
    if top is not None and bottom is not None:
        return _format_number(top), _format_number(bottom)

    depth_text = _first_text(row, ["prof", "profundidade", "camada_nome"])
    if depth_text:
        match = re.search(r"(\d+(?:[.,]\d+)?)\s*[-/]\s*(\d+(?:[.,]\d+)?)", depth_text)
        if match:
            return (
                _format_number(_to_float(match.group(1))),
                _format_number(_to_float(match.group(2))),
            )
    return "", ""


def _depth_from_text(value: object) -> str | None:
    text = _clean_text(value)
    match = re.search(r"(\d+(?:[.,]\d+)?)\s*-+\s*(\d+(?:[.,]\d+)?)", text)
    if not match:
        return None
    return (
        f"{_format_number(_to_float(match.group(1)))}-"
        f"{_format_number(_to_float(match.group(2)))}"
    )


def _carbon_content(row: dict[str, str]) -> ParsedCarbon | None:
    for key, value in row.items():
        if not _is_carbon_key(key):
            continue
        numeric = _to_optional_float(value)
        if numeric is None:
            continue
        if "%" in key or key.startswith("cot"):
            numeric *= 10.0
        return ParsedCarbon(value_g_kg=numeric, source_column=key)
    return None


def _target_measurements(row: dict[str, str]) -> list[ParsedTargetValue]:
    carbon = _carbon_content(row)
    if carbon is not None:
        return [
            ParsedTargetValue(
                target_kind="carbon_content",
                target_value=carbon.value_g_kg,
                target_unit="g/kg",
                carbon_content_g_kg=carbon.value_g_kg,
            )
        ]

    measurements: list[ParsedTargetValue] = []
    for key, value in row.items():
        numeric = _to_optional_float(value)
        if numeric is None:
            continue
        if _is_carbon_fraction_key(key):
            measurements.append(
                ParsedTargetValue(
                    target_kind=f"carbon_fraction:{_target_slug(key)}",
                    target_value=numeric,
                    target_unit="g/kg",
                    derivation_method="reported_carbon_fraction_by_source",
                )
            )
        elif _is_carbon_stock_key(key):
            measurements.append(
                ParsedTargetValue(
                    target_kind=f"carbon_stock:{_target_slug(key)}",
                    target_value=numeric,
                    target_unit="Mg/ha",
                    carbon_stock_mg_ha=numeric,
                    derivation_method="reported_layer_stock_by_source",
                )
            )
        elif _is_organic_matter_key(key):
            measurements.append(
                ParsedTargetValue(
                    target_kind="organic_matter_proxy",
                    target_value=numeric,
                    target_unit=_organic_matter_unit(key),
                    derivation_method="reported_organic_matter_not_converted_to_carbon",
                    quality_flag="proxy_target",
                )
            )
    return measurements


def _is_carbon_key(key: str) -> bool:
    if _is_carbon_fraction_key(key) or _is_carbon_stock_key(key):
        return False
    if key in {"c", "carbono", "carbono organico", "cot"}:
        return True
    if "carbono" in key or "organic carbon" in key:
        return True
    return key.startswith("cot ") or key.startswith("c ")


def _is_carbon_fraction_key(key: str) -> bool:
    parts = key.split()
    return len(parts) >= 2 and parts[0] == "c" and parts[1] in {
        "af",
        "ah",
        "hum",
        "mop",
        "mom",
    }


def _is_carbon_stock_key(key: str) -> bool:
    return key.startswith("estc ") or key.startswith("est c ") or (
        key.startswith("estoque") and " c " in f" {key} "
    )


def _is_organic_matter_key(key: str) -> bool:
    parts = key.split()
    return "materia organica" in key or (bool(parts) and parts[0] == "mo")


def _organic_matter_unit(key: str) -> str:
    if "dm3" in key or "dm 3" in key:
        return "g/dm3"
    if "kg" in key:
        return "g/kg"
    return ""


def _target_slug(key: str) -> str:
    unit_tokens = {"g", "kg", "mg", "ha", "dm", "dm3", "cm", "m", "3", "1"}
    return "_".join(part for part in key.split() if part not in unit_tokens)


def _bulk_density(row: dict[str, str]) -> float | None:
    return _first_float(
        row,
        [
            "densidade do solo mg m 3",
            "densidade do solo",
            "densidade",
            "ds g cm 3",
            "ds anel vol g cm 3",
            "dsi_cilindro",
        ],
    )


def _coarse_fraction(row: dict[str, str]) -> float | None:
    value = _first_float(row, ["pedregosidade", "cascalho_olho"])
    if value is None:
        return None
    if value > 1:
        return value / 100.0
    return value


def _sample_period(row: dict[str, str]) -> tuple[str, str, str, str]:
    raw = _first_text(
        row,
        ["data do evento", "observacao_data", "data", "ano coleta", "ano"],
    )
    if not raw:
        return "", "", "", ""
    iso_match = re.match(r"^(\d{4})-\d{2}-\d{2}$", raw)
    if iso_match:
        year = normalize_year(iso_match.group(1))
        return raw, year, year, year
    range_match = re.search(r"(\d{4})\D+(\d{4})", raw)
    if range_match:
        start = normalize_year(range_match.group(1))
        end = normalize_year(range_match.group(2))
        return "", start if start == end else "", start, end
    year = normalize_year(raw)
    return "", year, year, year


def _event_id(row: dict[str, str]) -> str:
    return _first_text(row, ["id do evento", "identificacao do evento", "observacao_id"])


def _event_keys(row: dict[str, str]) -> list[str]:
    keys = []
    event_id = _event_id(row)
    if event_id:
        keys.append(event_id)
    use = _first_text(row, ["uso da terra", "uso atual da terra"])
    trench = _first_float(row, ["trincheira"])
    if use and trench is not None:
        keys.append(f"{use} {_format_number(trench)}")
    return keys


def _first_text(row: dict[str, str], keys: Iterable[str]) -> str:
    for key in keys:
        value = row.get(_normalize_key(key), "")
        if value and _normalize_key(value) not in NA_VALUES:
            return value
    return ""


def _first_float(row: dict[str, str], keys: Iterable[str]) -> float | None:
    for key in keys:
        value = _to_optional_float(row.get(_normalize_key(key), ""))
        if value is not None:
            return value
    return None


def _first_coordinate(row: dict[str, str], keys: Iterable[str]) -> float | None:
    for key in keys:
        raw_value = row.get(_normalize_key(key), "")
        value = _to_optional_float(raw_value)
        if value is not None:
            return value
        value = _dms_text_coordinate(raw_value)
        if value is not None:
            return value
    return None


def _to_optional_float(value: object) -> float | None:
    text = _clean_text(value)
    if not text or _normalize_key(text) in NA_VALUES:
        return None
    try:
        numeric = float(text.replace(",", "."))
    except ValueError:
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _to_float(value: object) -> float:
    numeric = _to_optional_float(value)
    return 0.0 if numeric is None else numeric


def _format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _dms_coordinate(row: dict[str, str], prefix: str) -> float | None:
    degrees = _first_float(row, [f"{prefix} grau"])
    minutes = _first_float(row, [f"{prefix} min"]) or 0.0
    seconds = _first_float(row, [f"{prefix} seg"]) or 0.0
    hemisphere = _first_text(row, [f"{prefix} hem"]).upper()
    if degrees is None:
        return None
    value = abs(degrees) + minutes / 60.0 + seconds / 3600.0
    if hemisphere in {"S", "O", "W"} or degrees < 0:
        value *= -1
    return value


def _dms_text_coordinate(value: object) -> float | None:
    text = _clean_text(value).upper().replace(",", ".")
    if not text:
        return None
    hemisphere_match = re.search(r"\b([NSEWO])\b", text)
    if not hemisphere_match:
        return None
    hemisphere = hemisphere_match.group(1)
    numbers = [float(match) for match in re.findall(r"\d+(?:\.\d+)?", text)]
    if not numbers:
        return None
    degrees = numbers[0]
    minutes = numbers[1] if len(numbers) > 1 else 0.0
    seconds = numbers[2] if len(numbers) > 2 else 0.0
    coordinate = degrees + minutes / 60.0 + seconds / 3600.0
    if hemisphere in {"S", "W", "O"} or text.strip().startswith("-"):
        coordinate *= -1
    return coordinate


def _sheet_by_name(sheets: dict[str, str], expected: str) -> tuple[str, str] | None:
    for sheet_name, sheet_text in sheets.items():
        normalized = _normalize_key(sheet_name)
        if normalized == expected:
            return sheet_name, sheet_text
    for sheet_name, sheet_text in sheets.items():
        normalized = _normalize_key(sheet_name)
        if expected in normalized and not _is_auxiliary_sheet(sheet_name):
            return sheet_name, sheet_text
    return None


def _is_auxiliary_sheet(sheet_name: str) -> bool:
    normalized = _normalize_key(sheet_name)
    return any(
        term in normalized
        for term in ("identificacao", "metodo", "validacao", "dicionario", "admin")
    )


def _clean_text(value: object) -> str:
    return str(value or "").strip().strip('"').strip()


def _normalize_key(value: object) -> str:
    text = _clean_text(value).lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-z0-9%]+", " ", text)
    return " ".join(text.split())
