from __future__ import annotations

import zipfile
from dataclasses import replace
from html import unescape
from io import BytesIO
from typing import Any
from urllib.parse import quote

from carbono_solo.collector.io import fetch_bytes, fetch_json, fetch_text
from carbono_solo.collector.models import (
    CATALOG_ONLY,
    FAILED,
    IMPLEMENTED,
    InventoryRecord,
    SourceResult,
)
from carbono_solo.collector.sources.soildata_tabular import (
    parse_paired_event_layer_tables,
    parse_tabular_targets,
    parse_xlsx_targets,
)


SEARCH_URL = "https://repositorio.soildata.mapbiomas.org/api/search?q=carbono&subtree=soildata&type=dataset"
DATASET_API_URL_TEMPLATE = "https://repositorio.soildata.mapbiomas.org/api/datasets/:persistentId/?persistentId={persistent_id}"
DATAFILE_URL_TEMPLATE = "https://repositorio.soildata.mapbiomas.org/api/access/datafile/{file_id}"
SOURCE_ID = "soildata_search"
SOURCE_NAME = "SoilData Dataverse Search"
MAX_TARGET_FILE_BYTES = 25_000_000


def _score_item(item: dict[str, Any]) -> str:
    text = " ".join(
        str(item.get(key, ""))
        for key in ("name", "description", "keywords", "subjects")
    ).lower()
    strong_terms = (
        "estoque",
        "carbon stock",
        "densidade",
        "bulk density",
        "profundidade",
    )
    if "carbono" in text and any(term in text for term in strong_terms):
        return "high"
    if "carbono" in text or "organic carbon" in text:
        return "medium"
    return "low"


def records_from_search_payload(
    payload: dict[str, Any],
    retrieved_at: str,
) -> list[InventoryRecord]:
    data = payload.get("data", {})
    if not isinstance(data, dict):
        return []
    items = data.get("items", [])
    if not isinstance(items, list):
        return []

    return [
        InventoryRecord(
            source_id=SOURCE_ID,
            source_name=SOURCE_NAME,
            dataset_id=str(item.get("global_id", "")),
            title=unescape(str(item.get("name", ""))).strip(),
            url=str(item.get("url", "")),
            country_scope=_country_scope(item),
            access_type="dataverse_search_result",
            brazil_filter="native_brazil_repository",
            candidate_target_fields="described in dataset metadata",
            target_fit_score=_score_item(item),
            implementation_status=CATALOG_ONLY,
            license="",
            notes=_description(item)[:500],
            retrieved_at=retrieved_at,
        )
        for item in items
        if isinstance(item, dict)
    ]


def collect_soildata_search(
    retrieved_at: str,
    use_network: bool = True,
) -> SourceResult:
    if not use_network:
        return SourceResult(inventory=[], targets=[])

    try:
        payload = fetch_json(SEARCH_URL)
        if not isinstance(payload, dict):
            raise ValueError("SoilData search response was not an object.")
        inventory = records_from_search_payload(payload, retrieved_at)
        updated_inventory: list[InventoryRecord] = []
        targets = []
        for record in inventory:
            try:
                dataset_targets = _collect_dataset_targets(
                    dataset_id=record.dataset_id,
                    retrieved_at=retrieved_at,
                )
                targets.extend(dataset_targets)
                if dataset_targets:
                    record = replace(
                        record,
                        implementation_status=IMPLEMENTED,
                        candidate_target_fields=(
                            "coordinates, layer depth, carbon content, bulk density "
                            "when available"
                        ),
                        notes=_append_note(
                            record.notes,
                            f"Parsed {len(dataset_targets)} target rows from SoilData tabular files.",
                        )[:500],
                    )
            except Exception as exc:
                record = replace(
                    record,
                    notes=_append_note(record.notes, f"Ingestion error: {exc}")[:500],
                )
            updated_inventory.append(record)
        return SourceResult(
            inventory=updated_inventory,
            targets=targets,
        )
    except Exception as exc:
        return SourceResult(
            inventory=[_failed_inventory(retrieved_at, str(exc))],
            targets=[],
        )


def _description(item: dict[str, Any]) -> str:
    return (
        unescape(str(item.get("description", "")))
        .replace("\r", " ")
        .replace("\n", " ")
        .strip()
    )


def _country_scope(item: dict[str, Any]) -> str:
    geographies = item.get("geographicCoverage") or []
    if not isinstance(geographies, list):
        return "Brazil"

    countries = sorted(
        {
            str(geo.get("country", "")).strip().strip(",")
            for geo in geographies
            if isinstance(geo, dict) and str(geo.get("country", "")).strip()
        }
    )
    return "; ".join(countries) if countries else "Brazil"


def _failed_inventory(retrieved_at: str, notes: str) -> InventoryRecord:
    return InventoryRecord(
        source_id=SOURCE_ID,
        source_name=SOURCE_NAME,
        dataset_id="soildata_search_carbono",
        title="SoilData search for carbono",
        url=SEARCH_URL,
        country_scope="Brazil",
        access_type="dataverse_search",
        brazil_filter="native_brazil_repository",
        candidate_target_fields="carbono, estoque, densidade, profundidade",
        target_fit_score="high",
        implementation_status=FAILED,
        license="",
        notes=notes,
        retrieved_at=retrieved_at,
    )


def _collect_dataset_targets(dataset_id: str, retrieved_at: str) -> list:
    if not dataset_id:
        return []

    metadata = fetch_json(
        DATASET_API_URL_TEMPLATE.format(persistent_id=quote(dataset_id, safe=""))
    )
    if not isinstance(metadata, dict):
        raise ValueError(f"Dataset metadata response was not an object: {dataset_id}")

    files = _target_files(metadata)
    targets = []
    used_file_ids: set[str] = set()

    text_files = [file_info for file_info in files if file_info["kind"] == "text"]
    event_file = _first_file_matching(text_files, ("evento",))
    layer_file = _first_file_matching(text_files, ("camada", "layer"))
    if event_file and layer_file and event_file["id"] != layer_file["id"]:
        event_text = fetch_text(_datafile_url(event_file["id"]))
        layer_text = fetch_text(_datafile_url(layer_file["id"]))
        targets.extend(
            parse_paired_event_layer_tables(
                event_text,
                layer_text,
                dataset_id=dataset_id,
                event_file_id=event_file["id"],
                layer_file_id=layer_file["id"],
                retrieved_at=retrieved_at,
            )
        )
        used_file_ids.update({event_file["id"], layer_file["id"]})

    for file_info in files:
        if file_info["id"] in used_file_ids:
            continue
        kind = file_info["kind"]
        if kind == "text" and _is_auxiliary_table(file_info["filename"]):
            continue
        if kind == "text":
            targets.extend(
                parse_tabular_targets(
                    fetch_text(_datafile_url(file_info["id"])),
                    dataset_id=dataset_id,
                    file_id=file_info["id"],
                    file_name=file_info["filename"],
                    retrieved_at=retrieved_at,
                )
            )
        elif kind == "xlsx":
            targets.extend(
                parse_xlsx_targets(
                    fetch_bytes(_datafile_url(file_info["id"])),
                    dataset_id=dataset_id,
                    file_id=file_info["id"],
                    file_name=file_info["filename"],
                    retrieved_at=retrieved_at,
                )
            )
        elif kind == "zip":
            targets.extend(
                _parse_zip_targets(
                    fetch_bytes(_datafile_url(file_info["id"])),
                    dataset_id=dataset_id,
                    file_id=file_info["id"],
                    file_name=file_info["filename"],
                    retrieved_at=retrieved_at,
                )
            )

    return targets


def _target_files(metadata: dict[str, Any]) -> list[dict[str, str]]:
    latest = metadata.get("data", {}).get("latestVersion", {})
    files = latest.get("files", []) if isinstance(latest, dict) else []
    if not isinstance(files, list):
        return []

    parsed_files = []
    for file_info in files:
        if not isinstance(file_info, dict):
            continue
        data_file = file_info.get("dataFile", {})
        if not isinstance(data_file, dict):
            continue
        file_id = str(data_file.get("id", "")).strip()
        if not file_id:
            continue
        filename = str(data_file.get("filename") or file_info.get("label") or "")
        content_type = str(data_file.get("contentType", "")).lower()
        filesize = _safe_int(data_file.get("filesize", 0))
        if filesize and filesize > MAX_TARGET_FILE_BYTES:
            continue
        kind = _file_kind(filename, content_type)
        if not kind:
            continue
        parsed_files.append(
            {
                "id": file_id,
                "filename": filename,
                "content_type": content_type,
                "kind": kind,
            }
        )
    return parsed_files


def _file_kind(
    filename: str,
    content_type: str,
) -> str:
    lowered = filename.lower()
    if lowered.endswith((".tab", ".tsv", ".txt", ".csv")):
        return "text"
    if lowered.endswith(".xlsx"):
        return "xlsx"
    if lowered.endswith(".zip"):
        return "zip"
    if content_type in {
        "text/tab-separated-values",
        "text/plain",
        "text/csv",
        "application/csv",
    }:
        return "text"
    if content_type in {
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/xlsx",
    }:
        return "xlsx"
    if content_type in {
        "application/zip",
        "application/x-zip-compressed",
    }:
        return "zip"
    return ""


def _parse_zip_targets(
    data: bytes,
    *,
    dataset_id: str,
    file_id: str,
    file_name: str,
    retrieved_at: str,
) -> list:
    targets = []
    text_entries: dict[str, str] = {}

    try:
        with zipfile.ZipFile(BytesIO(data)) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                member_name = member.filename
                kind = _file_kind(member_name, "")
                if kind == "xlsx":
                    targets.extend(
                        parse_xlsx_targets(
                            archive.read(member),
                            dataset_id=dataset_id,
                            file_id=f"{file_id}:{member_name}",
                            file_name=f"{file_name}/{member_name}",
                            retrieved_at=retrieved_at,
                        )
                    )
                elif kind == "text":
                    text_entries[member_name] = archive.read(member).decode(
                        "utf-8-sig",
                        errors="replace",
                    )
    except zipfile.BadZipFile as exc:
        raise ValueError(f"Invalid ZIP file {file_name}") from exc

    used_entries: set[str] = set()
    event_name = _first_name_matching(text_entries, ("evento",))
    layer_name = _first_name_matching(text_entries, ("camada", "layer"))
    if event_name and layer_name and event_name != layer_name:
        targets.extend(
            parse_paired_event_layer_tables(
                text_entries[event_name],
                text_entries[layer_name],
                dataset_id=dataset_id,
                event_file_id=f"{file_id}:{event_name}",
                layer_file_id=f"{file_id}:{layer_name}",
                retrieved_at=retrieved_at,
            )
        )
        used_entries.update({event_name, layer_name})

    for member_name, text in text_entries.items():
        if member_name in used_entries:
            continue
        if _is_auxiliary_table(member_name):
            continue
        targets.extend(
            parse_tabular_targets(
                text,
                dataset_id=dataset_id,
                file_id=f"{file_id}:{member_name}",
                file_name=f"{file_name}/{member_name}",
                retrieved_at=retrieved_at,
            )
        )
    return targets


def _first_file_matching(
    files: list[dict[str, str]],
    terms: tuple[str, ...],
) -> dict[str, str] | None:
    for file_info in files:
        name = file_info["filename"].lower()
        if any(term in name for term in terms):
            return file_info
    return None


def _first_name_matching(
    entries: dict[str, str],
    terms: tuple[str, ...],
) -> str | None:
    for name in entries:
        lowered = name.lower()
        if any(term in lowered for term in terms):
            return name
    return None


def _is_auxiliary_table(filename: str) -> bool:
    lowered = filename.lower()
    return any(
        term in lowered
        for term in (
            "evento",
            "metodo",
            "historico",
            "identificacao",
            "identificação",
        )
    )


def _datafile_url(file_id: str) -> str:
    return DATAFILE_URL_TEMPLATE.format(file_id=file_id)


def _safe_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _append_note(current: str, addition: str) -> str:
    if not current:
        return addition
    if not addition:
        return current
    return f"{current} {addition}"
