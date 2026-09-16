from __future__ import annotations

from dataclasses import replace
from decimal import Decimal, InvalidOperation
from typing import Dict, Iterable, List, Tuple

from carbono_solo.collector.geo import approx_duplicate_coord_key, normalize_year
from carbono_solo.collector.models import TargetRecord


SOURCE_PRIORITY = {
    "hybras": 5,
    "mapbiomas": 10,
    "soildata": 20,
    "febr": 30,
    "bdsolos": 40,
    "wosis": 50,
    "israd": 60,
}


DuplicateKey = Tuple[str, str, str, str, str]
HybrasOverlapKey = Tuple[str, str, str, str, str]


def _canonical_depth_value(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        numeric = Decimal(text)
    except InvalidOperation:
        return text
    if not numeric.is_finite():
        return text
    if numeric == numeric.to_integral_value():
        return str(int(numeric))
    return format(numeric.normalize(), "f").rstrip("0").rstrip(".")


def _canonical_year(value: str) -> str:
    return normalize_year(value)


def _period_key(record: TargetRecord) -> str:
    sample_date = str(record.sample_date or "").strip()
    if sample_date:
        return sample_date
    sample_year = _canonical_year(record.sample_year)
    if sample_year:
        return sample_year
    period_start = _canonical_year(record.sample_period_start)
    period_end = _canonical_year(record.sample_period_end)
    if period_start or period_end:
        period_start = period_start or str(record.sample_period_start or "").strip()
        period_end = period_end or str(record.sample_period_end or "").strip()
        return f"{period_start}-{period_end}"
    return f"{record.sample_period_start}-{record.sample_period_end}"


def duplicate_key(record: TargetRecord) -> DuplicateKey:
    coord_key = approx_duplicate_coord_key(record.latitude, record.longitude)
    depth_key = (
        f"{_canonical_depth_value(record.depth_top_cm)}-"
        f"{_canonical_depth_value(record.depth_bottom_cm)}"
    )
    return (coord_key, _period_key(record), depth_key, record.target_kind, record.target_unit)


def _source_priority(source_id: str) -> int:
    normalized = source_id.lower()
    for source_family, priority in SOURCE_PRIORITY.items():
        if normalized == source_family or normalized.startswith(f"{source_family}_"):
            return priority
    return 999


def _priority(record: TargetRecord) -> Tuple[int, str]:
    return (_source_priority(record.source_id), record.unified_observation_id)


def _hybras_overlap_key(record: TargetRecord) -> HybrasOverlapKey | None:
    is_direct_hybras = record.source_id.lower() == "hybras_sgb"
    is_wosis_hybras = (
        record.source_id.lower() == "wosis_brazil"
        and record.source_observation_id.upper().startswith("BR-HYBRAS:")
    )
    if not is_direct_hybras and not is_wosis_hybras:
        return None
    try:
        target_value = Decimal(str(record.target_value)).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None
    coord_key = approx_duplicate_coord_key(record.latitude, record.longitude)
    depth_key = (
        f"{_canonical_depth_value(record.depth_top_cm)}-"
        f"{_canonical_depth_value(record.depth_bottom_cm)}"
    )
    return (
        coord_key,
        depth_key,
        record.target_kind,
        record.target_unit,
        format(target_value, "f"),
    )


def deduplicate_targets(records: Iterable[TargetRecord]) -> List[TargetRecord]:
    ordered = list(records)
    groups: Dict[DuplicateKey, List[int]] = {}
    for index, record in enumerate(ordered):
        groups.setdefault(duplicate_key(record), []).append(index)

    parents = list(range(len(ordered)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for indexes in groups.values():
        for index in indexes[1:]:
            union(indexes[0], index)

    direct_hybras: Dict[HybrasOverlapKey, List[int]] = {}
    for index, record in enumerate(ordered):
        if record.source_id.lower() != "hybras_sgb":
            continue
        overlap_key = _hybras_overlap_key(record)
        if overlap_key is not None:
            direct_hybras.setdefault(overlap_key, []).append(index)

    for index, record in enumerate(ordered):
        if record.source_id.lower() != "wosis_brazil":
            continue
        overlap_key = _hybras_overlap_key(record)
        for hybras_index in direct_hybras.get(overlap_key, []):
            union(index, hybras_index)

    components: Dict[int, List[int]] = {}
    for index in range(len(ordered)):
        components.setdefault(find(index), []).append(index)

    duplicate_components = [indexes for indexes in components.values() if len(indexes) > 1]
    duplicate_components.sort(
        key=lambda indexes: min(
            (duplicate_key(ordered[index]), ordered[index].unified_observation_id)
            for index in indexes
        )
    )
    group_ids = {
        find(indexes[0]): f"dup-{group_index:06d}"
        for group_index, indexes in enumerate(duplicate_components, start=1)
    }

    resolved = list(ordered)
    for root, indexes in components.items():
        if len(indexes) == 1:
            index = indexes[0]
            resolved[index] = replace(
                ordered[index],
                duplicate_group_id="",
                duplicate_status="active",
                duplicate_resolution="unique",
            )
            continue

        winner_index = min(indexes, key=lambda index: _priority(ordered[index]))
        winner = ordered[winner_index]
        group_id = group_ids[find(root)]
        for index in indexes:
            record = ordered[index]
            if index == winner_index:
                resolved[index] = replace(
                    record,
                    duplicate_group_id=group_id,
                    duplicate_status="active",
                    duplicate_resolution=f"kept_by_priority:{winner.source_id}",
                )
            else:
                resolved[index] = replace(
                    record,
                    duplicate_group_id=group_id,
                    duplicate_status="duplicate",
                    duplicate_resolution=f"removed_in_favor_of:{winner.source_id}",
                )

    return resolved
