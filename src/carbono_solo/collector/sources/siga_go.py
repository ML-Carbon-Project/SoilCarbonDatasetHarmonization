from __future__ import annotations

from pathlib import Path
from typing import Any

from carbono_solo.collector.io import fetch_json
from carbono_solo.collector.models import (
    CATALOG_ONLY,
    FAILED,
    InventoryRecord,
    SourceResult,
)


SIGA_RESOURCE_URL = "https://siga.meioambiente.go.gov.br/api/v2/resources/423"
LOCAL_SHAPEFILE_DIR = Path("pedo_area_GO_raster")
SOURCE_ID = "siga_go_solos_ibge"
SOURCE_NAME = "SIGA-GO Solos - IBGE"
DEFAULT_DATASET_ID = "geonode:solos_ibge_recorte20"
DEFAULT_TITLE = "Solos - IBGE"
DEFAULT_DETAIL_URL = "https://siga.meioambiente.go.gov.br/catalogue/#/dataset/423"
CANDIDATE_FIELDS = "soil order, texture, horizon, relief, stoniness"


def summarize_local_shapefile(base_dir: Path = LOCAL_SHAPEFILE_DIR) -> str:
    shp = base_dir / "pedo_area_uf_go.shp"
    dbf = base_dir / "pedo_area_uf_go.dbf"
    prj = base_dir / "pedo_area_uf_go.prj"
    if not (shp.exists() and dbf.exists() and prj.exists()):
        return "Local shapefile not found."

    prj_text = prj.read_text(encoding="utf-8", errors="replace").strip()
    return (
        f"Local shapefile found: shp_bytes={shp.stat().st_size}, "
        f"dbf_bytes={dbf.stat().st_size}, prj={prj_text[:120]}"
    )


def inventory_from_siga_payload(
    payload: dict[str, Any],
    retrieved_at: str,
    local_note: str | None = None,
) -> InventoryRecord:
    resource = payload.get("resource", {})
    if not isinstance(resource, dict):
        resource = {}

    abstract = _clean_text(resource.get("abstract", ""))
    notes = " ".join(part for part in (abstract, local_note or "") if part).strip()
    license_info = resource.get("license") or {}
    license_name = (
        _text_or_default(license_info.get("identifier"), "")
        if isinstance(license_info, dict)
        else _text_or_default(license_info, "")
    )

    return InventoryRecord(
        source_id=SOURCE_ID,
        source_name=SOURCE_NAME,
        dataset_id=_text_or_default(resource.get("alternate"), DEFAULT_DATASET_ID),
        title=_text_or_default(resource.get("title"), DEFAULT_TITLE),
        url=_text_or_default(resource.get("detail_url"), DEFAULT_DETAIL_URL),
        country_scope="Brazil: Goias",
        access_type="geonode_api_wfs",
        brazil_filter="native_goias_layer",
        candidate_target_fields=CANDIDATE_FIELDS,
        target_fit_score="low",
        implementation_status=CATALOG_ONLY,
        license=license_name,
        notes=notes,
        retrieved_at=retrieved_at,
    )


def collect_siga_go(
    retrieved_at: str,
    use_network: bool = True,
    local_dir: Path = LOCAL_SHAPEFILE_DIR,
) -> SourceResult:
    local_note = summarize_local_shapefile(local_dir)
    if not use_network:
        return SourceResult(
            inventory=[
                inventory_from_siga_payload(
                    {"resource": {}},
                    retrieved_at,
                    local_note=local_note,
                )
            ],
            targets=[],
        )

    try:
        payload = fetch_json(SIGA_RESOURCE_URL)
        if not isinstance(payload, dict):
            raise ValueError("SIGA-GO resource response was not an object.")
        return SourceResult(
            inventory=[
                inventory_from_siga_payload(
                    payload,
                    retrieved_at,
                    local_note=local_note,
                )
            ],
            targets=[],
        )
    except Exception as exc:
        return SourceResult(
            inventory=[_failed_inventory(retrieved_at, f"{exc}; {local_note}")],
            targets=[],
        )


def _failed_inventory(retrieved_at: str, notes: str) -> InventoryRecord:
    return InventoryRecord(
        source_id=SOURCE_ID,
        source_name=SOURCE_NAME,
        dataset_id=DEFAULT_DATASET_ID,
        title=DEFAULT_TITLE,
        url=DEFAULT_DETAIL_URL,
        country_scope="Brazil: Goias",
        access_type="geonode_api_wfs",
        brazil_filter="native_goias_layer",
        candidate_target_fields=CANDIDATE_FIELDS,
        target_fit_score="low",
        implementation_status=FAILED,
        license="",
        notes=notes,
        retrieved_at=retrieved_at,
    )


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).replace("\r", " ").replace("\n", " ").strip()


def _text_or_default(value: object, default: str) -> str:
    text = _clean_text(value)
    return text if text else default
