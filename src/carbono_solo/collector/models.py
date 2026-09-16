from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List


IMPLEMENTED = "implemented"
CATALOG_ONLY = "catalog_only"
MANUAL_OR_API_PENDING = "manual_or_api_pending"
FAILED = "failed"
SKIPPED_NO_TARGET_FIELDS = "skipped_no_target_fields"

INVENTORY_FIELDS: List[str] = [
    "source_id",
    "source_name",
    "dataset_id",
    "title",
    "url",
    "country_scope",
    "access_type",
    "brazil_filter",
    "candidate_target_fields",
    "target_fit_score",
    "implementation_status",
    "license",
    "notes",
    "retrieved_at",
]

TARGET_FIELDS: List[str] = [
    "unified_observation_id",
    "source_id",
    "dataset_id",
    "source_observation_id",
    "latitude",
    "longitude",
    "coord_x",
    "coord_y",
    "crs",
    "country",
    "state",
    "sample_date",
    "sample_year",
    "sample_period_start",
    "sample_period_end",
    "depth_top_cm",
    "depth_bottom_cm",
    "layer_thickness_cm",
    "carbon_content_g_kg",
    "carbon_analysis_method",
    "carbon_stock_kg_m2",
    "carbon_stock_mg_ha",
    "bulk_density_g_cm3",
    "coarse_fragments_fraction",
    "carbon_stock_00_05",
    "carbon_stock_05_15",
    "carbon_stock_15_30",
    "carbon_stock_30_60",
    "carbon_stock_60_100",
    "target_kind",
    "target_value",
    "target_unit",
    "derivation_method",
    "quality_flag",
    "duplicate_group_id",
    "duplicate_status",
    "duplicate_resolution",
]


@dataclass(frozen=True)
class InventoryRecord:
    source_id: str
    source_name: str
    dataset_id: str
    title: str
    url: str
    country_scope: str
    access_type: str
    brazil_filter: str
    candidate_target_fields: str
    target_fit_score: str
    implementation_status: str
    license: str
    notes: str
    retrieved_at: str

    def to_row(self) -> Dict[str, object]:
        values = asdict(self)
        return {field: values[field] for field in INVENTORY_FIELDS}


@dataclass(frozen=True)
class TargetRecord:
    unified_observation_id: str
    source_id: str
    dataset_id: str
    source_observation_id: str
    latitude: float
    longitude: float
    coord_x: float
    coord_y: float
    crs: str
    country: str
    state: str
    sample_date: str
    sample_year: str
    sample_period_start: str
    sample_period_end: str
    depth_top_cm: str
    depth_bottom_cm: str
    layer_thickness_cm: str
    carbon_content_g_kg: str
    carbon_analysis_method: str = field(
        default="N\u00e3o informado",
        kw_only=True,
    )
    carbon_stock_kg_m2: str
    carbon_stock_mg_ha: str
    bulk_density_g_cm3: str
    coarse_fragments_fraction: str
    carbon_stock_00_05: str
    carbon_stock_05_15: str
    carbon_stock_15_30: str
    carbon_stock_30_60: str
    carbon_stock_60_100: str
    target_kind: str
    target_value: str
    target_unit: str
    derivation_method: str
    quality_flag: str
    duplicate_group_id: str
    duplicate_status: str
    duplicate_resolution: str

    def to_row(self) -> Dict[str, object]:
        values = asdict(self)
        return {field: values[field] for field in TARGET_FIELDS}


@dataclass(frozen=True)
class SourceResult:
    inventory: List[InventoryRecord]
    targets: List[TargetRecord]
