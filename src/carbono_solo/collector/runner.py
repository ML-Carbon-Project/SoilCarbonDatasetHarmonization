from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Protocol

from carbono_solo.collector.dedup import deduplicate_targets
from carbono_solo.collector.io import write_csv
from carbono_solo.collector.models import (
    INVENTORY_FIELDS,
    TARGET_FIELDS,
    InventoryRecord,
    SourceResult,
    TargetRecord,
)
from carbono_solo.collector.sources.bdsolos import collect_bdsolos_embrapa
from carbono_solo.collector.sources.hybras import collect_hybras
from carbono_solo.collector.sources.israd import collect_israd_brazil
from carbono_solo.collector.sources.mapbiomas_soc import collect_mapbiomas_soc
from carbono_solo.collector.sources.siga_go import collect_siga_go
from carbono_solo.collector.sources.soildata import collect_soildata_search
from carbono_solo.collector.sources.static_catalog import (
    collect_static_catalog as _collect_static_catalog,
)
from carbono_solo.collector.sources.wosis import collect_wosis_brazil


class Adapter(Protocol):
    def __call__(self, *, retrieved_at: str, use_network: bool) -> SourceResult:
        ...


def collect_static_catalog(*, retrieved_at: str, use_network: bool = True) -> SourceResult:
    return _collect_static_catalog(retrieved_at=retrieved_at)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00",
        "Z",
    )


def default_adapters() -> tuple[Adapter, ...]:
    return (
        collect_static_catalog,
        collect_soildata_search,
        collect_mapbiomas_soc,
        collect_bdsolos_embrapa,
        collect_hybras,
        collect_wosis_brazil,
        collect_israd_brazil,
        collect_siga_go,
    )


def _run_adapter(
    adapter: Adapter,
    retrieved_at: str,
    use_network: bool,
) -> SourceResult:
    return adapter(retrieved_at=retrieved_at, use_network=use_network)


def run_collection(
    output_dir: Path,
    retrieved_at: str,
    use_network: bool,
    adapters: Iterable[Adapter] | None = None,
) -> dict[str, int]:
    inventory_records: list[InventoryRecord] = []
    target_records: list[TargetRecord] = []

    for adapter in adapters if adapters is not None else default_adapters():
        result = _run_adapter(adapter, retrieved_at, use_network)
        inventory_records.extend(result.inventory)
        target_records.extend(result.targets)

    deduplicated_targets = deduplicate_targets(target_records)
    active_targets = [
        target for target in deduplicated_targets if target.duplicate_status == "active"
    ]
    duplicate_targets = [
        target
        for target in deduplicated_targets
        if target.duplicate_status == "duplicate"
    ]

    write_csv(
        output_dir / "dataset_inventory_brasil.csv",
        INVENTORY_FIELDS,
        (record.to_row() for record in inventory_records),
    )
    write_csv(
        output_dir / "soil_targets_brasil.csv",
        TARGET_FIELDS,
        (record.to_row() for record in active_targets),
    )
    write_csv(
        output_dir / "soil_targets_duplicates_brasil.csv",
        TARGET_FIELDS,
        (record.to_row() for record in duplicate_targets),
    )

    return {
        "inventory_records": len(inventory_records),
        "active_targets": len(active_targets),
        "duplicate_targets": len(duplicate_targets),
    }
