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

SCIENTIFIC_FIELDS = (
    "layer_thickness_cm",
    "carbon_content_g_kg",
    "carbon_stock_kg_m2",
    "carbon_stock_mg_ha",
    "bulk_density_g_cm3",
    "coarse_fragments_fraction",
    "carbon_stock_00_05",
    "carbon_stock_05_15",
    "carbon_stock_15_30",
    "carbon_stock_30_60",
    "carbon_stock_60_100",
    "target_value",
    "carbon_analysis_method",
)

NUMERIC_SCIENTIFIC_FIELDS = frozenset(SCIENTIFIC_FIELDS) - {
    "carbon_analysis_method"
}

DuplicateKey = Tuple[str, str, str, str, str]
HybrasLineageKey = Tuple[str, str, str, str]
HybrasOverlapKey = Tuple[str, str, str, str, str]
ScientificFingerprint = Tuple[str, ...]
SourceIdentity = Tuple[str, str, str]


def _canonical_decimal(value: object) -> str:
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


def _canonical_depth_value(value: str) -> str:
    return _canonical_decimal(value)


def _canonical_year(value: str) -> str:
    return normalize_year(value)


def _canonical_method(value: object) -> str:
    text = " ".join(str(value or "").strip().casefold().split())
    if text in {"", "nao informado", "não informado", "not reported"}:
        return ""
    return text


def _canonical_scientific_value(field: str, value: object) -> str:
    if field == "carbon_analysis_method":
        return _canonical_method(value)
    if field in NUMERIC_SCIENTIFIC_FIELDS:
        return _canonical_decimal(value)
    return " ".join(str(value or "").strip().casefold().split())


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


def _metadata_score(record: TargetRecord) -> int:
    fields = (
        "source_observation_id",
        "sample_date",
        "sample_year",
        "sample_period_start",
        "sample_period_end",
        "state",
        "carbon_content_g_kg",
        "carbon_stock_kg_m2",
        "carbon_stock_mg_ha",
        "bulk_density_g_cm3",
        "coarse_fragments_fraction",
        "carbon_analysis_method",
    )
    score = sum(bool(str(getattr(record, field) or "").strip()) for field in fields)
    if _canonical_method(record.carbon_analysis_method):
        score += 1
    return score


def _priority(record: TargetRecord) -> Tuple[int, int, str, str, str, str]:
    return (
        -_metadata_score(record),
        _source_priority(record.source_id),
        record.source_id,
        record.dataset_id,
        record.source_observation_id,
        record.unified_observation_id,
    )


def _scientific_fingerprint(record: TargetRecord) -> ScientificFingerprint:
    return tuple(
        _canonical_scientific_value(field, getattr(record, field))
        for field in SCIENTIFIC_FIELDS
    )


def _source_identity(record: TargetRecord) -> SourceIdentity | None:
    observation_id = str(record.source_observation_id or "").strip()
    if not observation_id:
        return None
    return (record.source_id, record.dataset_id, observation_id)


def _same_identity_compatible(left: TargetRecord, right: TargetRecord) -> bool:
    identity = _source_identity(left)
    if identity is None or identity != _source_identity(right):
        return False
    for field in SCIENTIFIC_FIELDS:
        left_value = _canonical_scientific_value(field, getattr(left, field))
        right_value = _canonical_scientific_value(field, getattr(right, field))
        if left_value and right_value and left_value != right_value:
            return False
    return True


def _is_direct_hybras(record: TargetRecord) -> bool:
    return record.source_id.lower() == "hybras_sgb"


def _is_wosis_hybras(record: TargetRecord) -> bool:
    return (
        record.source_id.lower() == "wosis_brazil"
        and record.source_observation_id.upper().startswith("BR-HYBRAS:")
    )


def _hybras_lineage_key(record: TargetRecord) -> HybrasLineageKey | None:
    if not _is_direct_hybras(record) and not _is_wosis_hybras(record):
        return None
    coord_key = approx_duplicate_coord_key(record.latitude, record.longitude)
    depth_key = (
        f"{_canonical_depth_value(record.depth_top_cm)}-"
        f"{_canonical_depth_value(record.depth_bottom_cm)}"
    )
    return (coord_key, depth_key, record.target_kind, record.target_unit)


def _hybras_overlap_key(record: TargetRecord) -> HybrasOverlapKey | None:
    lineage_key = _hybras_lineage_key(record)
    if lineage_key is None:
        return None
    try:
        target_value = Decimal(str(record.target_value)).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None
    return (*lineage_key, format(target_value, "f"))


def _confirmed_duplicate_reason(left: TargetRecord, right: TargetRecord) -> str:
    if (
        (_is_direct_hybras(left) and _is_wosis_hybras(right))
        or (_is_direct_hybras(right) and _is_wosis_hybras(left))
    ) and _hybras_overlap_key(left) == _hybras_overlap_key(right):
        return "documented_republication"
    if duplicate_key(left) != duplicate_key(right):
        return ""
    if _same_identity_compatible(left, right):
        return "shared_source_observation_id"
    if _scientific_fingerprint(left) == _scientific_fingerprint(right):
        return "exact_scientific_match"
    return ""


def _component_reason(records: list[TargetRecord]) -> str:
    identities: Dict[SourceIdentity, int] = {}
    for record in records:
        identity = _source_identity(record)
        if identity is not None:
            identities[identity] = identities.get(identity, 0) + 1
    if any(count > 1 for count in identities.values()):
        return "identifier_conflict"
    target_values = {
        _canonical_decimal(record.target_value)
        for record in records
        if _canonical_decimal(record.target_value)
    }
    if len(target_values) > 1:
        return "discordant_target_value"
    return "discordant_scientific_fields"


def _uniquify_active_observation_ids(
    records: list[TargetRecord],
) -> list[TargetRecord]:
    active_by_id: Dict[str, List[int]] = {}
    for index, record in enumerate(records):
        if record.duplicate_status == "active":
            active_by_id.setdefault(record.unified_observation_id, []).append(index)

    resolved = list(records)
    for observation_id, indexes in active_by_id.items():
        if len(indexes) < 2:
            continue
        ordered_indexes = sorted(
            indexes,
            key=lambda index: (
                f"{records[index].latitude:.10f}",
                f"{records[index].longitude:.10f}",
                _scientific_fingerprint(records[index]),
                records[index].dataset_id,
                records[index].source_observation_id,
                index,
            ),
        )
        coordinate_ids: Dict[str, int] = {}
        for index in ordered_indexes:
            record = records[index]
            coordinate_id = (
                f"{observation_id}:coord:"
                f"{record.latitude:.7f},{record.longitude:.7f}"
            )
            variant = coordinate_ids.get(coordinate_id, 0) + 1
            coordinate_ids[coordinate_id] = variant
            unique_id = (
                coordinate_id if variant == 1 else f"{coordinate_id}:variant:{variant}"
            )
            resolved[index] = replace(record, unified_observation_id=unique_id)
    return resolved


def deduplicate_targets(records: Iterable[TargetRecord]) -> List[TargetRecord]:
    ordered = list(records)
    candidate_parents = list(range(len(ordered)))

    def find(parents: list[int], index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(parents: list[int], left: int, right: int) -> None:
        left_root = find(parents, left)
        right_root = find(parents, right)
        if left_root != right_root:
            parents[right_root] = left_root

    duplicate_groups: Dict[DuplicateKey, List[int]] = {}
    for index, record in enumerate(ordered):
        duplicate_groups.setdefault(duplicate_key(record), []).append(index)
    for indexes in duplicate_groups.values():
        for index in indexes[1:]:
            union(candidate_parents, indexes[0], index)

    hybras_lineages: Dict[HybrasLineageKey, List[int]] = {}
    for index, record in enumerate(ordered):
        lineage_key = _hybras_lineage_key(record)
        if lineage_key is not None:
            hybras_lineages.setdefault(lineage_key, []).append(index)
    for indexes in hybras_lineages.values():
        has_direct = any(_is_direct_hybras(ordered[index]) for index in indexes)
        has_wosis = any(_is_wosis_hybras(ordered[index]) for index in indexes)
        if not has_direct or not has_wosis:
            continue
        for index in indexes[1:]:
            union(candidate_parents, indexes[0], index)

    candidate_components: Dict[int, List[int]] = {}
    for index in range(len(ordered)):
        candidate_components.setdefault(find(candidate_parents, index), []).append(index)

    overlap_components = [
        indexes for indexes in candidate_components.values() if len(indexes) > 1
    ]
    overlap_components.sort(
        key=lambda indexes: min(
            (duplicate_key(ordered[index]), ordered[index].unified_observation_id)
            for index in indexes
        )
    )
    group_ids = {
        find(candidate_parents, indexes[0]): f"dup-{group_index:06d}"
        for group_index, indexes in enumerate(overlap_components, start=1)
    }

    resolved = list(ordered)
    for root, indexes in candidate_components.items():
        if len(indexes) == 1:
            index = indexes[0]
            resolved[index] = replace(
                ordered[index],
                duplicate_group_id="",
                duplicate_status="active",
                duplicate_resolution="unique",
            )
            continue

        confirmed_parents = {index: index for index in indexes}
        evidence: Dict[Tuple[int, int], str] = {}

        def confirmed_find(index: int) -> int:
            while confirmed_parents[index] != index:
                confirmed_parents[index] = confirmed_parents[confirmed_parents[index]]
                index = confirmed_parents[index]
            return index

        def confirmed_union(left: int, right: int, reason: str) -> None:
            left_root = confirmed_find(left)
            right_root = confirmed_find(right)
            if left_root != right_root:
                confirmed_parents[right_root] = left_root
            evidence[tuple(sorted((left, right)))] = reason

        for left_position, left_index in enumerate(indexes):
            for right_index in indexes[left_position + 1 :]:
                reason = _confirmed_duplicate_reason(
                    ordered[left_index],
                    ordered[right_index],
                )
                if reason:
                    confirmed_union(left_index, right_index, reason)

        confirmed_components: Dict[int, List[int]] = {}
        for index in indexes:
            confirmed_components.setdefault(confirmed_find(index), []).append(index)

        winners: Dict[int, int] = {}
        for confirmed_root, confirmed_indexes in confirmed_components.items():
            winners[confirmed_root] = min(
                confirmed_indexes,
                key=lambda index: _priority(ordered[index]),
            )

        active_indexes = sorted(winners.values())
        ambiguous = len(active_indexes) > 1
        active_records = [ordered[index] for index in active_indexes]
        ambiguity_reason = _component_reason(active_records) if ambiguous else ""
        group_id = group_ids[find(candidate_parents, root)]

        for confirmed_root, confirmed_indexes in confirmed_components.items():
            winner_index = winners[confirmed_root]
            winner = ordered[winner_index]
            cluster_evidence = {
                reason
                for pair, reason in evidence.items()
                if pair[0] in confirmed_indexes and pair[1] in confirmed_indexes
            }
            proof = (
                "documented_republication"
                if "documented_republication" in cluster_evidence
                else "shared_source_observation_id"
                if "shared_source_observation_id" in cluster_evidence
                else "exact_scientific_match"
            )
            for index in confirmed_indexes:
                record = ordered[index]
                if index == winner_index:
                    resolution = (
                        f"retained_ambiguous_overlap:{ambiguity_reason}"
                        if ambiguous
                        else f"kept_confirmed_duplicate:{proof}"
                    )
                    resolved[index] = replace(
                        record,
                        duplicate_group_id=group_id,
                        duplicate_status="active",
                        duplicate_resolution=resolution,
                    )
                else:
                    resolved[index] = replace(
                        record,
                        duplicate_group_id=group_id,
                        duplicate_status="duplicate",
                        duplicate_resolution=(
                            f"removed_confirmed_duplicate:{proof}:"
                            f"{winner.source_id}"
                        ),
                    )

    return _uniquify_active_observation_ids(resolved)
