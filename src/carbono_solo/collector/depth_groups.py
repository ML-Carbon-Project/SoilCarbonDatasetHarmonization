from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from mpspline import mpspline

from carbono_solo.collector.io import write_csv


GROUP_FIELDS = [
    "SOURCE_ID",
    "DATASET_ID",
    "LATITUDE",
    "LONGITUDE",
    "CRS",
    "COUNTRY",
    "STATE",
    "SAMPLE_DATE",
    "SAMPLE_YEAR",
]

STOCK_DEPTH_FIELDS = [
    "Stock_00_05",
    "Stock_05_15",
    "Stock_15_30",
    "Stock_30_60",
    "Stock_60_100",
]

OUTPUT_FIELDS = [*GROUP_FIELDS, *STOCK_DEPTH_FIELDS]

DEPTH_BUCKETS = [
    (0.0, 5.0, "Stock_00_05", "COVERAGE_00_05"),
    (5.0, 15.0, "Stock_05_15", "COVERAGE_05_15"),
    (15.0, 30.0, "Stock_15_30", "COVERAGE_15_30"),
    (30.0, 60.0, "Stock_30_60", "COVERAGE_30_60"),
    (60.0, 100.0, "Stock_60_100", "COVERAGE_60_100"),
]

AUDIT_FIELDS = [
    *OUTPUT_FIELDS,
    "PROFILE_LAYER_COUNT_INPUT",
    "PROFILE_LAYER_COUNT_USED",
    "EXACT_DUPLICATE_LAYER_COUNT",
    "OVERLAPPING_INPUT_LAYERS",
    "DEPTH_HARMONIZATION_METHOD",
    "SPLINE_SUPPORTED",
    "SPLINE_REASON",
    "SPLINE_LAMBDA",
    "SPLINE_RESCALE_FACTOR",
    "SPLINE_PRE_RESCALE_MASS_BALANCE_ERROR_PCT",
    "STANDARD_DEPTH_REFERENCE_STOCK_Mg_ha",
    "STANDARD_DEPTH_OUTPUT_STOCK_Mg_ha",
    "MASS_BALANCE_ERROR_PCT",
    "EXTRAPOLATED",
    *[bucket[3] for bucket in DEPTH_BUCKETS],
    *[f"OVERLAP_REFERENCE_{bucket[2]}" for bucket in DEPTH_BUCKETS],
]

_TOLERANCE = 1e-8


@dataclass(frozen=True)
class _Layer:
    top: float
    bottom: float
    stock: float
    source_index: int
    calculation_method: str
    target_kind: str

    @property
    def thickness(self) -> float:
        return self.bottom - self.top

    @property
    def stock_density(self) -> float:
        return self.stock / self.thickness


@dataclass(frozen=True)
class _Segment:
    top: float
    bottom: float
    stock_density: float
    source_index: int


@dataclass(frozen=True)
class _ProfileResult:
    output_row: dict[str, str] | None
    audit_row: dict[str, str]
    method: str
    assigned_values: int
    full_bands: int
    partial_bands: int
    uncovered_bands: int


def group_stock_by_depth_csv(
    input_path: str | Path,
    output_path: str | Path,
    audit_output_path: str | Path | None = None,
    *,
    spline_lambda: float = 0.1,
    max_spline_mass_error_pct: float = 10.0,
    use_splines: bool = True,
) -> dict[str, int | float]:
    rows = _read_csv(input_path)
    fields, grouped_rows, audit_rows, summary = harmonize_stock_depth_rows(
        rows,
        spline_lambda=spline_lambda,
        max_spline_mass_error_pct=max_spline_mass_error_pct,
        use_splines=use_splines,
    )
    write_csv(output_path, fields, grouped_rows)
    if audit_output_path is not None:
        write_csv(audit_output_path, AUDIT_FIELDS, audit_rows)

    return {
        "input_rows": len(rows),
        "output_rows": len(grouped_rows),
        "audit_profiles": len(audit_rows),
        **summary,
    }


def group_stock_by_depth_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    spline_lambda: float = 0.1,
    max_spline_mass_error_pct: float = 10.0,
    use_splines: bool = True,
) -> tuple[list[str], list[dict[str, str]]]:
    fields, grouped_rows, _audit_rows, _summary = harmonize_stock_depth_rows(
        rows,
        spline_lambda=spline_lambda,
        max_spline_mass_error_pct=max_spline_mass_error_pct,
        use_splines=use_splines,
    )
    return fields, grouped_rows


def harmonize_stock_depth_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    spline_lambda: float = 0.1,
    max_spline_mass_error_pct: float = 10.0,
    use_splines: bool = True,
) -> tuple[
    list[str],
    list[dict[str, str]],
    list[dict[str, str]],
    dict[str, int | float],
]:
    if spline_lambda < 0:
        raise ValueError("spline_lambda must be non-negative")
    if max_spline_mass_error_pct < 0:
        raise ValueError("max_spline_mass_error_pct must be non-negative")

    row_list = list(rows)
    grouped: dict[tuple[str, ...], list[_Layer]] = {}
    skipped_without_target = 0
    skipped_without_valid_depth = 0

    for source_index, row in enumerate(row_list):
        target_value = _to_float(row.get("TARGET_VALUE"))
        if target_value is None or target_value < 0:
            skipped_without_target += 1
            continue

        top = _to_float(row.get("DEPTH_TOP_CM"))
        bottom = _to_float(row.get("DEPTH_BOTTOM_CM"))
        if top is None or bottom is None or top < 0 or bottom <= top:
            skipped_without_valid_depth += 1
            continue
        if bottom <= 0 or top >= 100:
            skipped_without_valid_depth += 1
            continue

        key = tuple(_clean(row.get(field)) for field in GROUP_FIELDS)
        grouped.setdefault(key, []).append(
            _Layer(
                top=top,
                bottom=bottom,
                stock=target_value,
                source_index=source_index,
                calculation_method=_clean(row.get("STOCK_CALCULATION_METHOD")),
                target_kind=_clean(row.get("ORIGINAL_TARGET_KIND")),
            )
        )

    output_rows: list[dict[str, str]] = []
    audit_rows: list[dict[str, str]] = []
    method_counts = {
        "profiles_equal_area_spline": 0,
        "profiles_mass_preserving_overlap": 0,
    }
    assigned_values = 0
    full_bands = 0
    partial_bands = 0
    uncovered_bands = 0

    for key, layers in grouped.items():
        base = {field: value for field, value in zip(GROUP_FIELDS, key)}
        result = _harmonize_profile(
            base,
            layers,
            spline_lambda=spline_lambda,
            max_spline_mass_error_pct=max_spline_mass_error_pct,
            use_splines=use_splines,
        )
        audit_rows.append(result.audit_row)
        if result.output_row is not None:
            output_rows.append(result.output_row)
        method_counts[f"profiles_{result.method}"] += 1
        assigned_values += result.assigned_values
        full_bands += result.full_bands
        partial_bands += result.partial_bands
        uncovered_bands += result.uncovered_bands

    summary: dict[str, int | float] = {
        "assigned_stock_values": assigned_values,
        "skipped_rows_without_target": skipped_without_target,
        "skipped_rows_without_valid_depth": skipped_without_valid_depth,
        "fully_covered_profile_bands": full_bands,
        "partially_covered_profile_bands": partial_bands,
        "uncovered_profile_bands": uncovered_bands,
        **method_counts,
    }
    return OUTPUT_FIELDS.copy(), output_rows, audit_rows, summary


def _harmonize_profile(
    base: Mapping[str, str],
    input_layers: list[_Layer],
    *,
    spline_lambda: float,
    max_spline_mass_error_pct: float,
    use_splines: bool,
) -> _ProfileResult:
    layers, exact_duplicates = _deduplicate_exact_layers(input_layers)
    layers.sort(key=lambda layer: (layer.top, layer.bottom, layer.source_index))
    has_overlap = _has_overlap(layers)
    segments = _select_non_overlapping_segments(layers)
    reference_stocks, coverage = _overlap_stocks(segments)

    spline_supported, spline_reason = _spline_support(layers, use_splines)
    method = "mass_preserving_overlap"
    output_stocks = reference_stocks.copy()
    rescale_factor: float | None = None
    pre_balance_error: float | None = None

    if spline_supported:
        try:
            spline_stocks = _spline_stocks(layers, coverage, spline_lambda)
            reference_total = _sum_present(reference_stocks)
            spline_total = _sum_present(spline_stocks)
            pre_balance_error = _relative_error_pct(spline_total, reference_total)
            if reference_total <= _TOLERANCE:
                spline_supported = False
                spline_reason = "zero_reference_stock"
            elif not math.isfinite(pre_balance_error):
                spline_supported = False
                spline_reason = "non_finite_mass_balance"
            elif abs(pre_balance_error) > max_spline_mass_error_pct:
                spline_supported = False
                spline_reason = "mass_balance_guard"
            elif spline_total > _TOLERANCE:
                rescale_factor = reference_total / spline_total
                spline_stocks = {
                    field: value * rescale_factor if value is not None else None
                    for field, value in spline_stocks.items()
                }
            else:
                raise ValueError("spline returned no positive mass")
            if spline_supported:
                output_stocks = spline_stocks
                method = "equal_area_spline"
                spline_reason = "applied"
        except (ArithmeticError, KeyError, TypeError, ValueError) as exc:
            spline_supported = False
            spline_reason = f"spline_error:{type(exc).__name__}"

    reference_total = _sum_present(reference_stocks)
    output_total = _sum_present(output_stocks)
    balance_error = _relative_error_pct(output_total, reference_total)

    output_row = {field: str(base[field]) for field in GROUP_FIELDS}
    for field in STOCK_DEPTH_FIELDS:
        value = output_stocks[field]
        output_row[field] = _format_float(value) if value is not None else ""

    assigned = sum(bool(output_row[field]) for field in STOCK_DEPTH_FIELDS)
    final_output_row = output_row if assigned else None

    audit_row = output_row.copy()
    audit_row.update(
        {
            "PROFILE_LAYER_COUNT_INPUT": str(len(input_layers)),
            "PROFILE_LAYER_COUNT_USED": str(len(layers)),
            "EXACT_DUPLICATE_LAYER_COUNT": str(exact_duplicates),
            "OVERLAPPING_INPUT_LAYERS": _format_bool(has_overlap),
            "DEPTH_HARMONIZATION_METHOD": method,
            "SPLINE_SUPPORTED": _format_bool(spline_supported),
            "SPLINE_REASON": spline_reason,
            "SPLINE_LAMBDA": _format_float(spline_lambda) if use_splines else "",
            "SPLINE_RESCALE_FACTOR": _format_optional_float(rescale_factor),
            "SPLINE_PRE_RESCALE_MASS_BALANCE_ERROR_PCT": _format_optional_float(
                pre_balance_error
            ),
            "STANDARD_DEPTH_REFERENCE_STOCK_Mg_ha": _format_float(reference_total),
            "STANDARD_DEPTH_OUTPUT_STOCK_Mg_ha": _format_float(output_total),
            "MASS_BALANCE_ERROR_PCT": _format_float(balance_error),
            "EXTRAPOLATED": "false",
        }
    )
    for _top, _bottom, _field, coverage_field in DEPTH_BUCKETS:
        audit_row[coverage_field] = _format_float(coverage[coverage_field])
    for _top, _bottom, stock_field, _coverage_field in DEPTH_BUCKETS:
        audit_row[f"OVERLAP_REFERENCE_{stock_field}"] = _format_optional_float(
            reference_stocks[stock_field]
        )

    full = sum(value >= 1.0 - _TOLERANCE for value in coverage.values())
    partial = sum(_TOLERANCE < value < 1.0 - _TOLERANCE for value in coverage.values())
    uncovered = len(DEPTH_BUCKETS) - full - partial
    return _ProfileResult(
        output_row=final_output_row,
        audit_row=audit_row,
        method=method,
        assigned_values=assigned,
        full_bands=full,
        partial_bands=partial,
        uncovered_bands=uncovered,
    )


def _deduplicate_exact_layers(layers: list[_Layer]) -> tuple[list[_Layer], int]:
    by_interval: dict[tuple[float, float], list[_Layer]] = {}
    for layer in layers:
        key = (round(layer.top, 8), round(layer.bottom, 8))
        by_interval.setdefault(key, []).append(layer)

    selected = [min(candidates, key=_layer_priority) for candidates in by_interval.values()]
    return selected, len(layers) - len(selected)


def _layer_priority(layer: _Layer) -> tuple[int, int, int]:
    method = layer.calculation_method.lower()
    if method == "converted_from_kg_m2":
        method_rank = 0
    elif method.startswith("computed_from_carbon_content"):
        method_rank = 1
    elif method.startswith("existing_mg_ha"):
        method_rank = 2
    elif method.startswith("estimated_from_carbon_content"):
        method_rank = 3
    else:
        method_rank = 4

    kind = layer.target_kind.lower()
    if kind == "soc_stock_layer":
        kind_rank = 0
    elif kind in {"carbon_content", "total_carbon_content"}:
        kind_rank = 1
    elif kind == "organic_matter_proxy":
        kind_rank = 2
    else:
        kind_rank = 3
    return method_rank, kind_rank, layer.source_index


def _has_overlap(layers: list[_Layer]) -> bool:
    furthest_bottom = -math.inf
    for layer in layers:
        if layer.top < furthest_bottom - _TOLERANCE:
            return True
        furthest_bottom = max(furthest_bottom, layer.bottom)
    return False


def _has_gap(layers: list[_Layer]) -> bool:
    return any(
        current.bottom < following.top - _TOLERANCE
        for current, following in zip(layers, layers[1:])
    )


def _select_non_overlapping_segments(layers: list[_Layer]) -> list[_Segment]:
    boundaries = {0.0, 100.0}
    for layer in layers:
        clipped_top = max(0.0, layer.top)
        clipped_bottom = min(100.0, layer.bottom)
        if clipped_bottom > clipped_top:
            boundaries.update((clipped_top, clipped_bottom))
    boundaries.update(boundary for bucket in DEPTH_BUCKETS for boundary in bucket[:2])
    ordered = sorted(boundaries)

    segments: list[_Segment] = []
    for top, bottom in zip(ordered, ordered[1:]):
        if bottom <= top + _TOLERANCE:
            continue
        candidates = [
            layer
            for layer in layers
            if layer.top <= top + _TOLERANCE
            and layer.bottom >= bottom - _TOLERANCE
        ]
        if not candidates:
            continue
        selected = min(
            candidates,
            key=lambda layer: (layer.thickness, *_layer_priority(layer)),
        )
        segment = _Segment(top, bottom, selected.stock_density, selected.source_index)
        if (
            segments
            and segments[-1].source_index == segment.source_index
            and math.isclose(segments[-1].bottom, segment.top, abs_tol=_TOLERANCE)
        ):
            previous = segments[-1]
            segments[-1] = _Segment(
                previous.top,
                segment.bottom,
                previous.stock_density,
                previous.source_index,
            )
        else:
            segments.append(segment)
    return segments


def _overlap_stocks(
    segments: list[_Segment],
) -> tuple[dict[str, float | None], dict[str, float]]:
    stocks: dict[str, float | None] = {}
    coverage: dict[str, float] = {}
    for top, bottom, stock_field, coverage_field in DEPTH_BUCKETS:
        band_thickness = bottom - top
        covered = 0.0
        stock = 0.0
        for segment in segments:
            overlap = max(0.0, min(segment.bottom, bottom) - max(segment.top, top))
            if overlap <= 0:
                continue
            covered += overlap
            stock += segment.stock_density * overlap
        fraction = min(1.0, covered / band_thickness)
        coverage[coverage_field] = fraction
        stocks[stock_field] = stock if fraction >= 1.0 - _TOLERANCE else None
    return stocks, coverage


def _spline_support(layers: list[_Layer], use_splines: bool) -> tuple[bool, str]:
    if not use_splines:
        return False, "disabled"
    if len(layers) < 3:
        return False, "insufficient_layers"
    if _has_overlap(layers):
        return False, "overlapping_layers"
    if _has_gap(layers):
        return False, "gaps_between_layers"
    if any(
        not math.isclose(depth, round(depth), abs_tol=_TOLERANCE)
        for layer in layers
        for depth in (layer.top, layer.bottom)
    ):
        return False, "non_integer_depths"
    return True, "eligible"


def _spline_stocks(
    layers: list[_Layer],
    coverage: Mapping[str, float],
    spline_lambda: float,
) -> dict[str, float | None]:
    component = {
        "cokey": "profile",
        "horizons": [
            {
                "upper": int(round(layer.top)),
                "lower": int(round(min(layer.bottom, 100.0))),
                "stock_density": layer.stock_density,
            }
            for layer in layers
        ],
    }
    target_depths = [
        (int(top), int(bottom)) for top, bottom, _field, _coverage in DEPTH_BUCKETS
    ]
    result = mpspline(
        component,
        var_name="stock_density",
        target_depths=target_depths,
        lam=spline_lambda,
        vlow=0.0,
        vhigh=1000.0,
        strict=True,
        output_type="wide",
        mode="dcm",
    )
    if not isinstance(result, dict):
        raise TypeError("unexpected mpspline output")

    stocks: dict[str, float | None] = {}
    for top, bottom, stock_field, coverage_field in DEPTH_BUCKETS:
        if coverage[coverage_field] < 1.0 - _TOLERANCE:
            stocks[stock_field] = None
            continue
        value = _to_float(result.get(f"stock_density_{int(top)}_{int(bottom)}"))
        if value is None or value < 0:
            raise ValueError("invalid spline estimate")
        stocks[stock_field] = value * (bottom - top)
    return stocks


def _sum_present(stocks: Mapping[str, float | None]) -> float:
    return sum(value for value in stocks.values() if value is not None)


def _relative_error_pct(estimate: float, reference: float) -> float:
    if math.isclose(reference, 0.0, abs_tol=_TOLERANCE):
        return 0.0 if math.isclose(estimate, 0.0, abs_tol=_TOLERANCE) else math.inf
    return (estimate - reference) / reference * 100.0


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=";"))


def _to_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", ".")
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _clean(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _format_bool(value: bool) -> str:
    return "true" if value else "false"


def _format_optional_float(value: float | None) -> str:
    return "" if value is None else _format_float(value)


def _format_float(value: float) -> str:
    if math.isclose(value, 0.0, abs_tol=1e-12):
        return "0"
    return f"{value:.10f}".rstrip("0").rstrip(".")
