from __future__ import annotations

import csv
import json
import math
import os
import tempfile
import textwrap
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Mapping, Sequence

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "carbono_solo_matplotlib"),
)

import matplotlib
import numpy as np
import shapefile
from pyproj import Geod
from scipy import stats
from shapely.geometry import shape

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle


SOURCE_ORDER = [
    "mapbiomas_soc",
    "wosis_brazil",
    "bdsolos_embrapa",
    "soildata_dataset",
    "israd_brazil",
    "hybras_sgb",
]
SOURCE_LABELS = {
    "mapbiomas_soc": "MapBiomas Soil C3",
    "wosis_brazil": "WoSIS",
    "bdsolos_embrapa": "BD Solos / Embrapa",
    "soildata_dataset": "SoilData",
    "israd_brazil": "ISRaD",
    "hybras_sgb": "HYBRAS / SGB",
}
SOURCE_COLORS = {
    "mapbiomas_soc": "#087F5B",
    "wosis_brazil": "#3568A8",
    "bdsolos_embrapa": "#C44E52",
    "soildata_dataset": "#B07A12",
    "israd_brazil": "#7B5AA6",
    "hybras_sgb": "#4C6A64",
}
SOURCE_NODE_LABELS = {
    "mapbiomas_soc": "MapBiomas\nSoil C3",
    "wosis_brazil": "WoSIS",
    "bdsolos_embrapa": "BD Solos",
    "soildata_dataset": "SoilData",
    "israd_brazil": "ISRaD",
    "hybras_sgb": "HYBRAS",
}
BIOME_ORDER = [
    "Amazônia",
    "Cerrado",
    "Mata Atlântica",
    "Caatinga",
    "Pampa",
    "Pantanal",
]
BIOME_LABELS_EN = {
    "Amazônia": "Amazon",
    "Cerrado": "Cerrado",
    "Mata Atlântica": "Atlantic Forest",
    "Caatinga": "Caatinga",
    "Pampa": "Pampa",
    "Pantanal": "Pantanal",
}
STATE_CODES = {
    "11": "RO", "12": "AC", "13": "AM", "14": "RR", "15": "PA",
    "16": "AP", "17": "TO", "21": "MA", "22": "PI", "23": "CE",
    "24": "RN", "25": "PB", "26": "PE", "27": "AL", "28": "SE",
    "29": "BA", "31": "MG", "32": "ES", "33": "RJ", "35": "SP",
    "41": "PR", "42": "SC", "43": "RS", "50": "MS", "51": "MT",
    "52": "GO", "53": "DF",
}
DEPTH_ORDER = ["0-5", "5-15", "15-30", "30-60", "60-100", ">100"]
DEPTH_STOCK_BANDS = [
    ("0-5", "Stock_00_05", "COVERAGE_00_05"),
    ("5-15", "Stock_05_15", "COVERAGE_05_15"),
    ("15-30", "Stock_15_30", "COVERAGE_15_30"),
    ("30-60", "Stock_30_60", "COVERAGE_30_60"),
    ("60-100", "Stock_60_100", "COVERAGE_60_100"),
]
@dataclass(frozen=True)
class HarmonizedRow:
    source: str
    dataset: str
    latitude: float
    longitude: float
    state: str
    biome: str
    sample_year: str
    depth_top: float | None
    depth_bottom: float | None
    carbon: float
    method: str
    target_kind: str
    stock_method: str


def _float(value: str | None) -> float | None:
    text = (value or "").strip().replace(",", ".")
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _normalize(value: str) -> str:
    text = unicodedata.normalize("NFKD", value)
    return text.encode("ascii", "ignore").decode().casefold()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=";"))


def load_harmonized_rows(path: Path) -> list[HarmonizedRow]:
    rows: list[HarmonizedRow] = []
    for raw in _read_csv(path):
        latitude = _float(raw.get("LATITUDE"))
        longitude = _float(raw.get("LONGITUDE"))
        carbon = _float(raw.get("CARBON_CONTENT_G_KG"))
        if latitude is None or longitude is None or carbon is None:
            continue
        rows.append(
            HarmonizedRow(
                source=(raw.get("SOURCE_ID") or "").strip(),
                dataset=(raw.get("DATASET_ID") or "").strip(),
                latitude=latitude,
                longitude=longitude,
                state=(raw.get("STATE") or "").strip(),
                biome=(raw.get("BIOME") or "").strip(),
                sample_year=(raw.get("SAMPLE_YEAR") or "").strip(),
                depth_top=_float(raw.get("DEPTH_TOP_CM")),
                depth_bottom=_float(raw.get("DEPTH_BOTTOM_CM")),
                carbon=carbon,
                method=(raw.get("CARBON_ANALYSIS_METHOD") or "").strip(),
                target_kind=(raw.get("ORIGINAL_TARGET_KIND") or "").strip(),
                stock_method=(raw.get("STOCK_CALCULATION_METHOD") or "").strip(),
            )
        )
    return rows


def _line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def _target_origin(target_kind: str) -> str:
    if target_kind == "soc_stock_layer" or target_kind.startswith("carbon_stock"):
        return "Reported stock"
    if target_kind in {"carbon_content", "total_carbon_content"}:
        return "Measured carbon"
    return "Proxy or fraction"


def _stock_basis(stock_method: str) -> str:
    if stock_method.startswith("converted_from_") or stock_method.startswith("existing_"):
        return "Reported stock"
    if stock_method.startswith("computed_from_"):
        return "Observed bulk density"
    return "Estimated bulk density"


def _method_reported(method: str) -> bool:
    normalized = _normalize(method)
    return bool(method) and normalized not in {
        "nao informado",
        "nao se aplica - ponto pseudoamostral",
    }


def aggregate_source_points(rows: Iterable[HarmonizedRow]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float, float], list[HarmonizedRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.source, round(row.latitude, 5), round(row.longitude, 5))].append(row)
    points = []
    for (source, latitude, longitude), group in grouped.items():
        values = np.asarray([row.carbon for row in group], dtype=float)
        biomes = Counter(row.biome for row in group if row.biome)
        states = Counter(row.state for row in group if row.state)
        points.append(
            {
                "source": source,
                "latitude": latitude,
                "longitude": longitude,
                "carbon_median_g_kg": float(np.median(values)),
                "biome": biomes.most_common(1)[0][0] if biomes else "",
                "state": states.most_common(1)[0][0] if states else "",
                "layers": len(group),
            }
        )
    return points


def aggregate_locations_across_sources(
    points: Iterable[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[
        tuple[float, float], dict[str, list[float]]
    ] = defaultdict(lambda: defaultdict(list))
    for point in points:
        key = (float(point["latitude"]), float(point["longitude"]))
        grouped[key][str(point["source"])].append(
            float(point["carbon_median_g_kg"])
        )

    locations = []
    for (latitude, longitude), source_values in grouped.items():
        source_medians = np.asarray(
            [np.median(values) for values in source_values.values()],
            dtype=float,
        )
        locations.append(
            {
                "latitude": latitude,
                "longitude": longitude,
                "carbon_median_g_kg": float(np.median(source_medians)),
                "source_count": len(source_values),
                "sources": tuple(sorted(source_values)),
            }
        )
    return locations


def source_summary(
    rows: Sequence[HarmonizedRow],
    points: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    result = []
    for source in SOURCE_ORDER:
        source_rows = [row for row in rows if row.source == source]
        source_points = [point for point in points if point["source"] == source]
        values = np.asarray(
            [float(point["carbon_median_g_kg"]) for point in source_points],
            dtype=float,
        )
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        basis = Counter(_stock_basis(row.stock_method) for row in source_rows)
        result.append(
            {
                "source_id": source,
                "source_label": SOURCE_LABELS[source],
                "layers": len(source_rows),
                "unique_points": len(source_points),
                "datasets": len({row.dataset for row in source_rows}),
                "states": len({row.state for row in source_rows if row.state}),
                "biomes": len({row.biome for row in source_rows if row.biome}),
                "sample_year_reported_pct": 100
                * sum(bool(row.sample_year) for row in source_rows)
                / len(source_rows),
                "analysis_method_reported_pct": 100
                * sum(_method_reported(row.method) for row in source_rows)
                / len(source_rows),
                "median_g_kg": float(median),
                "q1_g_kg": float(q1),
                "q3_g_kg": float(q3),
                "reported_stock_rows": basis["Reported stock"],
                "observed_bulk_density_rows": basis["Observed bulk density"],
                "estimated_bulk_density_rows": basis["Estimated bulk density"],
            }
        )
    return result


def data_flow_summary(processed_dir: Path) -> dict[str, int | float]:
    active = _line_count(processed_dir / "soil_targets_brasil.csv")
    duplicates = _line_count(processed_dir / "soil_targets_duplicates_brasil.csv")
    harmonized = _line_count(processed_dir / "soil_targets_brasil_harmonized.csv")
    grouped = _line_count(processed_dir / "soil_target_Brasil_Stock_Group_Depth.csv")
    harmonization_audit = _read_csv(
        processed_dir / "soil_targets_brasil_harmonization_audit.csv"
    )
    organic_surface_density_rows = sum(
        row.get("EXCLUSION_REASON")
        == "organic_surface_layer_with_bulk_density_below_mineral_limit"
        for row in harmonization_audit
    )
    return {
        "collected_target_rows": active + duplicates,
        "duplicates_removed": duplicates,
        "active_after_deduplication": active,
        "organic_surface_density_rows_removed": organic_surface_density_rows,
        "territory_rows_removed": active - organic_surface_density_rows - harmonized,
        "harmonized_layers": harmonized,
        "grouped_point_rows": grouped,
    }


def overlap_rows(path: Path) -> list[dict[str, object]]:
    rows = []
    for row in _read_csv(path):
        rows.append(
            {
                **row,
                "shared_coordinates_5dp": int(row["shared_coordinates_5dp"] or 0),
                "near_identical_0_01_g_kg_share": _float(
                    row.get("near_identical_0_01_g_kg_share")
                ),
                "spearman_rho": _float(row.get("spearman_rho")),
            }
        )
    return rows


def _holm_adjust(p_values: Sequence[float | None]) -> list[float | None]:
    adjusted: list[float | None] = [None] * len(p_values)
    valid = sorted(
        (float(value), index)
        for index, value in enumerate(p_values)
        if value is not None and math.isfinite(float(value))
    )
    previous = 0.0
    total = len(valid)
    for rank, (value, index) in enumerate(valid):
        corrected = min(1.0, value * (total - rank))
        previous = max(previous, corrected)
        adjusted[index] = previous
    return adjusted


def _rank_biserial(differences: np.ndarray) -> float | None:
    nonzero = differences[np.abs(differences) > 1e-12]
    if len(nonzero) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero))
    denominator = float(ranks.sum())
    if denominator == 0:
        return None
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    return (positive - negative) / denominator


def matched_source_comparison(
    rows: Sequence[HarmonizedRow],
    *,
    require_same_year: bool,
) -> list[dict[str, object]]:
    layer_values: dict[str, dict[tuple[object, ...], list[float]]] = {
        source: defaultdict(list) for source in SOURCE_ORDER
    }
    for row in rows:
        if row.source not in layer_values or row.depth_top is None or row.depth_bottom is None:
            continue
        year = row.sample_year.strip()
        if require_same_year and not year:
            continue
        key: tuple[object, ...] = (
            round(row.latitude, 5),
            round(row.longitude, 5),
            round(row.depth_top, 3),
            round(row.depth_bottom, 3),
        )
        if require_same_year:
            key += (year,)
        layer_values[row.source][key].append(row.carbon)

    layer_medians = {
        source: {key: float(np.median(values)) for key, values in source_values.items()}
        for source, source_values in layer_values.items()
    }
    results: list[dict[str, object]] = []
    for source_a, source_b in combinations(SOURCE_ORDER, 2):
        shared_keys = sorted(set(layer_medians[source_a]) & set(layer_medians[source_b]))
        if not shared_keys:
            continue
        block_values: dict[tuple[object, ...], list[tuple[float, float]]] = defaultdict(list)
        for key in shared_keys:
            block = (key[0], key[1], key[4]) if require_same_year else (key[0], key[1])
            block_values[block].append(
                (layer_medians[source_a][key], layer_medians[source_b][key])
            )

        block_a = np.asarray(
            [np.median([pair[0] for pair in pairs]) for pairs in block_values.values()],
            dtype=float,
        )
        block_b = np.asarray(
            [np.median([pair[1] for pair in pairs]) for pairs in block_values.values()],
            dtype=float,
        )
        differences = np.asarray(
            [np.median([pair[0] - pair[1] for pair in pairs]) for pairs in block_values.values()],
            dtype=float,
        )
        scales = (np.abs(block_a) + np.abs(block_b)) / 2
        within_20 = np.where(
            scales > 1e-12,
            np.abs(differences) / scales <= 0.20,
            np.abs(differences) <= 0.01,
        )
        if len(differences) < 20:
            p_value = None
        elif np.all(np.abs(differences) <= 1e-12):
            p_value = 1.0
        else:
            p_value = float(
                stats.wilcoxon(
                    differences,
                    zero_method="pratt",
                    alternative="two-sided",
                    method="auto",
                ).pvalue
            )
        if len(block_a) >= 3 and np.ptp(block_a) > 0 and np.ptp(block_b) > 0:
            rho, rho_p = stats.spearmanr(block_a, block_b)
            rho = float(rho)
            rho_p = float(rho_p)
        else:
            rho = None
            rho_p = None
        results.append(
            {
                "source_a": source_a,
                "source_a_label": SOURCE_LABELS[source_a],
                "source_b": source_b,
                "source_b_label": SOURCE_LABELS[source_b],
                "pair_label": f"{SOURCE_NODE_LABELS[source_a].replace(chr(10), ' ')} vs {SOURCE_NODE_LABELS[source_b].replace(chr(10), ' ')}",
                "matching_rule": "coordinate_depth_year" if require_same_year else "coordinate_depth",
                "matched_layers": len(shared_keys),
                "matched_blocks": len(block_values),
                "median_difference_a_minus_b_g_kg": float(np.median(differences)),
                "q1_difference_g_kg": float(np.quantile(differences, 0.25)),
                "q3_difference_g_kg": float(np.quantile(differences, 0.75)),
                "median_absolute_difference_g_kg": float(np.median(np.abs(differences))),
                "near_identical_0_01_g_kg_share": float(np.mean(np.abs(differences) <= 0.01)),
                "within_20_percent_share": float(np.mean(within_20)),
                "spearman_rho": rho,
                "spearman_p": rho_p,
                "wilcoxon_p": p_value,
                "paired_rank_biserial": _rank_biserial(differences),
            }
        )

    adjusted = _holm_adjust([row["wilcoxon_p"] for row in results])
    for row, value in zip(results, adjusted):
        row["wilcoxon_p_holm"] = value
    return results


def _geometry_area_km2(geometry: object, geod: Geod) -> float:
    area, _perimeter = geod.geometry_area_perimeter(geometry)
    return abs(float(area)) / 1_000_000


def biome_areas(shapefile_path: Path) -> dict[str, float]:
    reader = shapefile.Reader(str(shapefile_path), encoding="utf-8")
    fields = [field[0] for field in reader.fields[1:]]
    name_index = fields.index("NM_BIOMA")
    geod = Geod(ellps="GRS80")
    areas: dict[str, float] = defaultdict(float)
    for shape_record in reader.iterShapeRecords():
        name = str(shape_record.record[name_index]).strip()
        if name in BIOME_ORDER:
            areas[name] += _geometry_area_km2(shape(shape_record.shape.__geo_interface__), geod)
    return dict(areas)


def state_areas(geojson: dict[str, object]) -> dict[str, float]:
    geod = Geod(ellps="GRS80")
    areas: dict[str, float] = {}
    for feature in geojson["features"]:
        code = str(feature["properties"]["codarea"])
        state = STATE_CODES.get(code)
        if state:
            areas[state] = _geometry_area_km2(shape(feature["geometry"]), geod)
    return areas


def representation_statistics(
    rows: Sequence[HarmonizedRow],
    biome_area: dict[str, float],
    state_area: dict[str, float],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    point_lookup: dict[tuple[float, float], HarmonizedRow] = {}
    for row in rows:
        point_lookup.setdefault((round(row.latitude, 5), round(row.longitude, 5)), row)
    biome_counts = Counter(row.biome for row in point_lookup.values() if row.biome)
    state_counts = Counter(row.state for row in point_lookup.values() if row.state)
    total_points = sum(biome_counts.values())
    total_biome_area = sum(biome_area.values())
    biome_rows = []
    for biome in BIOME_ORDER:
        point_share = biome_counts[biome] / total_points
        area_share = biome_area[biome] / total_biome_area
        biome_rows.append(
            {
                "biome": biome,
                "biome_label_en": BIOME_LABELS_EN[biome],
                "area_km2": biome_area[biome],
                "area_share_pct": 100 * area_share,
                "unique_points": biome_counts[biome],
                "point_share_pct": 100 * point_share,
                "representation_ratio": point_share / area_share,
                "points_per_1000_km2": biome_counts[biome] / biome_area[biome] * 1000,
            }
        )
    total_state_points = sum(state_counts.values())
    total_state_area = sum(state_area.values())
    state_rows = []
    for state in sorted(state_area):
        point_share = state_counts[state] / total_state_points
        area_share = state_area[state] / total_state_area
        ratio = point_share / area_share if area_share else math.nan
        state_rows.append(
            {
                "state": state,
                "area_km2": state_area[state],
                "area_share_pct": 100 * area_share,
                "unique_points": state_counts[state],
                "point_share_pct": 100 * point_share,
                "representation_ratio": ratio,
                "log2_representation_ratio": math.log2(ratio) if ratio > 0 else math.nan,
                "points_per_1000_km2": state_counts[state] / state_area[state] * 1000,
            }
        )
    return biome_rows, state_rows


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def _plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.labelsize": 9,
            "axes.edgecolor": "#A7B0B7",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "#E3E8EB",
            "grid.linewidth": 0.7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "legend.frameon": False,
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    if path.suffix.lower() == ".png":
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
        if path.stem == "graphical_abstract":
            fig.savefig(
                path.with_suffix(".tif"),
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
                pil_kwargs={"compression": "tiff_lzw"},
            )
    plt.close(fig)


def plot_workflow_provenance(
    flow: dict[str, int | float],
    source_rows: Sequence[dict[str, object]],
    overlaps: Sequence[dict[str, object]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 11.5))
    ax = axes[0, 0]
    ax.axis("off")
    boxes = [
        (
            0.02,
            0.40,
            "Target records\nbefore reconciliation",
            int(flow["collected_target_rows"]),
        ),
        (0.36, 0.40, "Retained after\nreconciliation", int(flow["active_after_deduplication"])),
        (0.70, 0.40, "Harmonized records", int(flow["harmonized_layers"])),
    ]
    for x, y, label, value in boxes:
        ax.add_patch(
            Rectangle(
                (x, y),
                0.26,
                0.28,
                transform=ax.transAxes,
                facecolor="#F4F7F7",
                edgecolor="#6C7A82",
                linewidth=1.2,
            )
        )
        ax.text(x + 0.13, y + 0.19, f"{value:,}", transform=ax.transAxes, ha="center", fontsize=18, fontweight="bold", color="#172126")
        ax.text(x + 0.13, y + 0.08, label, transform=ax.transAxes, ha="center", fontsize=9.5, color="#52616A")
    for left, right in ((0.28, 0.36), (0.62, 0.70)):
        ax.add_patch(FancyArrowPatch((left, 0.54), (right, 0.54), transform=ax.transAxes, arrowstyle="-|>", mutation_scale=14, color="#4F5B62"))
    ax.text(0.32, 0.70, f"-{int(flow['duplicates_removed']):,}\npotentially redundant records", transform=ax.transAxes, ha="center", color="#9A5E11", fontsize=9)
    ax.text(
        0.66,
        0.73,
        f"{int(flow['organic_surface_density_rows_removed']):,} organic-layer record exclusions\n"
        f"{int(flow['territory_rows_removed']):,} geographic exclusions",
        transform=ax.transAxes,
        ha="center",
        va="top",
        color="#9A5E11",
        fontsize=8.5,
    )
    ax.text(
        0.50,
        0.16,
        f"{int(flow['exact_coordinate_pairs']):,} exact coordinate pairs | "
        f"{int(flow['rounded_location_keys']):,} rounded locations (5 d.p.)",
        transform=ax.transAxes,
        ha="center",
        fontsize=10,
        fontweight="bold",
        color="#263238",
    )
    ax.set_title("a) Data flow and spatial indexing", loc="left", fontweight="bold")

    ax = axes[0, 1]
    labels = [str(row["source_label"]) for row in source_rows]
    layers = np.asarray([int(row["layers"]) for row in source_rows])
    points = np.asarray([int(row["unique_points"]) for row in source_rows])
    positions = np.arange(len(labels))
    width = 0.36
    ax.bar(positions - width / 2, layers, width, color="#3568A8", label="Records")
    ax.bar(positions + width / 2, points, width, color="#C58B2A", label="Rounded locations")
    ax.set_yscale("log")
    ax.set_xticks(positions, labels, rotation=28, ha="right")
    ax.set_ylabel("Count, logarithmic scale")
    ax.set_title("b) Contribution of each source", loc="left", fontweight="bold")
    ax.legend(ncol=2)
    ax.grid(axis="x", visible=False)

    ax = axes[1, 0]
    ax.set_aspect("equal")
    ax.axis("off")
    angles = np.linspace(math.pi / 2, math.pi / 2 + 2 * math.pi, len(SOURCE_ORDER), endpoint=False)
    positions_xy = {source: (math.cos(angle), math.sin(angle)) for source, angle in zip(SOURCE_ORDER, angles)}
    point_counts = {str(row["source_id"]): int(row["unique_points"]) for row in source_rows}
    positive_edges = [row for row in overlaps if int(row["shared_coordinates_5dp"]) > 0]
    max_edge = max((int(row["shared_coordinates_5dp"]) for row in positive_edges), default=1)
    for row in positive_edges:
        source_a = str(row["source_a"])
        source_b = str(row["source_b"])
        shared = int(row["shared_coordinates_5dp"])
        x1, y1 = positions_xy[source_a]
        x2, y2 = positions_xy[source_b]
        linewidth = 0.8 + 6 * math.sqrt(shared / max_edge)
        ax.plot([x1, x2], [y1, y2], color="#839198", linewidth=linewidth, alpha=0.55, zorder=1)
        ax.text((x1 + x2) / 2, (y1 + y2) / 2, f"{shared:,}", fontsize=7.5, ha="center", va="center", color="#263238", bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5}, zorder=3)
    max_points = max(point_counts.values())
    for source in SOURCE_ORDER:
        x, y = positions_xy[source]
        node_size = 1000 + 2100 * math.sqrt(point_counts[source] / max_points)
        ax.scatter(x, y, s=node_size, color=SOURCE_COLORS[source], edgecolor="white", linewidth=1.5, zorder=2)
        ax.text(x, y, SOURCE_NODE_LABELS[source], ha="center", va="center", color="white", fontsize=7.0, fontweight="bold", zorder=4)
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.4, 1.4)
    ax.set_title("c) Shared rounded locations between sources", loc="left", fontweight="bold")

    ax = axes[1, 1]
    categories = ["Reported stock", "Observed bulk density", "Estimated bulk density"]
    category_colors = ["#4C6A64", "#3568A8", "#C58B2A"]
    left = np.zeros(len(source_rows))
    y = np.arange(len(source_rows))
    totals = np.asarray([int(row["layers"]) for row in source_rows], dtype=float)
    for category, color in zip(categories, category_colors):
        field = {
            "Reported stock": "reported_stock_rows",
            "Observed bulk density": "observed_bulk_density_rows",
            "Estimated bulk density": "estimated_bulk_density_rows",
        }[category]
        values = 100 * np.asarray([int(row[field]) for row in source_rows]) / totals
        ax.barh(y, values, left=left, color=color, label=category, height=0.68)
        left += values
    ax.set_yticks(y, [str(row["source_label"]) for row in source_rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of harmonized records (%)")
    ax.set_title("d) Basis used to obtain carbon stock", loc="left", fontweight="bold")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.27), ncol=3, fontsize=8)
    ax.grid(axis="y", visible=False)

    fig.suptitle("Integration, redundancy, and provenance of the harmonized dataset", x=0.04, y=0.99, ha="left", fontsize=17, fontweight="bold")
    fig.text(0.04, 0.012, "Edges in panel c denote shared rounded locations and do not prove identical or independent sampling events. Stock derived with estimated bulk density carries additional model uncertainty.", fontsize=8.5, color="#59666F")
    fig.tight_layout(rect=(0.02, 0.05, 0.99, 0.95))
    _save(fig, output_path)


def _plot_geojson_boundaries(ax: plt.Axes, geojson: dict[str, object]) -> None:
    def draw_polygon(coordinates: Sequence[object]) -> None:
        exterior = np.asarray(coordinates[0], dtype=float)
        ax.plot(exterior[:, 0], exterior[:, 1], color="#9EA9AF", linewidth=0.35, zorder=1)

    for feature in geojson["features"]:
        geometry = feature["geometry"]
        if geometry["type"] == "Polygon":
            draw_polygon(geometry["coordinates"])
        elif geometry["type"] == "MultiPolygon":
            for polygon in geometry["coordinates"]:
                draw_polygon(polygon)


def plot_graphical_abstract(
    points: Sequence[dict[str, object]],
    geojson: dict[str, object],
    source_rows: Sequence[dict[str, object]],
    flow: dict[str, int | float],
    output_path: Path,
) -> None:
    # CATENA requests graphical abstracts at 13 x 5 cm (or the same ratio).
    fig = plt.figure(figsize=(13.28, 5.31))
    canvas = fig.add_axes([0, 0, 1, 1])
    canvas.set_axis_off()

    canvas.text(
        0.035,
        0.945,
        "Provenance-aware soil carbon evidence across Brazilian landscapes",
        fontsize=17,
        fontweight="bold",
        color="#18323B",
        va="top",
    )
    canvas.text(
        0.035,
        0.895,
        "Six public sources integrated before landscape-scale inference",
        fontsize=9.5,
        color="#52616A",
        va="top",
    )

    panels = [
        (0.03, 0.16, 0.24, 0.67, "INPUT", "Multi-source observations"),
        (0.31, 0.16, 0.36, 0.67, "WORKFLOW", "Harmonize, locate, and audit"),
        (0.71, 0.16, 0.26, 0.67, "EVIDENCE", "What the database supports"),
    ]
    for x, y, width, height, eyebrow, heading in panels:
        canvas.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.008,rounding_size=0.012",
                facecolor="#F8FAFB",
                edgecolor="#D7E0E4",
                linewidth=1.2,
            )
        )
        canvas.text(x + 0.018, y + height - 0.045, eyebrow, fontsize=7.0, fontweight="bold", color="#087F5B")
        canvas.text(x + 0.018, y + height - 0.085, heading, fontsize=9.5, fontweight="bold", color="#18323B")

    source_lookup = {row["source_id"]: row for row in source_rows}
    for index, source in enumerate(SOURCE_ORDER):
        y = 0.665 - index * 0.078
        canvas.add_patch(Rectangle((0.052, y - 0.016), 0.012, 0.041, facecolor=SOURCE_COLORS[source], edgecolor="none"))
        canvas.text(0.074, y + 0.011, SOURCE_LABELS[source], fontsize=6.8, fontweight="bold", color="#263238", va="center")
        canvas.text(
            0.247,
            y + 0.011,
            f"{int(source_lookup[source]['unique_points']):,} rounded locations",
            fontsize=6.2,
            color="#59666F",
            va="center",
            ha="right",
        )

    map_ax = fig.add_axes([0.355, 0.325, 0.27, 0.40])
    _plot_geojson_boundaries(map_ax, geojson)
    coordinates = np.asarray([(float(row["longitude"]), float(row["latitude"])) for row in points])
    map_ax.scatter(coordinates[:, 0], coordinates[:, 1], s=1.3, color="#087F5B", alpha=0.24, linewidths=0, rasterized=True)
    map_ax.set_xlim(-74.5, -32.5)
    map_ax.set_ylim(-34.5, 6.0)
    map_ax.set_aspect("equal", adjustable="box")
    map_ax.set_axis_off()

    workflow_steps = [
        ("1", "Brazilian land mask", "Remove marine and foreign points"),
        ("2", "Duplicate reconciliation", "Coordinate, period, depth, provenance"),
        ("3", "Target harmonization", "Units, mineral density limits, depth, stock"),
    ]
    for index, (number, label, detail) in enumerate(workflow_steps):
        x = 0.338 + index * 0.105
        canvas.text(x, 0.285, number, fontsize=10, fontweight="bold", color="white", ha="center", va="center", bbox={"boxstyle": "circle,pad=0.35", "facecolor": "#087F5B", "edgecolor": "none"})
        canvas.text(x, 0.251, label, fontsize=6.2, fontweight="bold", color="#263238", ha="center")
        canvas.text(x, 0.221, "\n".join(textwrap.wrap(detail, 18)), fontsize=5.5, color="#66757D", ha="center", va="top")

    evidence = [
        (f"{int(flow['harmonized_layers']):,}", "harmonized records", "#087F5B"),
        (f"{int(flow['exact_coordinate_pairs']):,}", "exact coordinate pairs", "#3568A8"),
        (f"{int(flow['rounded_location_keys']):,}", "rounded locations (5 d.p.)", "#B07A12"),
        ("47.0%", "stocks use estimated bulk density", "#C44E52"),
    ]
    for index, (value, label, color) in enumerate(evidence):
        y = 0.685 - index * 0.115
        canvas.text(0.742, y, value, fontsize=14, fontweight="bold", color=color, va="center")
        canvas.text(0.742, y - 0.038, label, fontsize=6.4, color="#52616A", va="center")

    canvas.text(0.742, 0.255, "MODELING IMPLICATION", fontsize=6.2, fontweight="bold", color="#087F5B")
    canvas.text(
        0.742,
        0.220,
        "Group profiles; validate by space and source;\npropagate density uncertainty.",
        fontsize=6.8,
        fontweight="bold",
        color="#263238",
        linespacing=1.35,
        va="top",
    )

    for start, end in [((0.273, 0.50), (0.307, 0.50)), ((0.673, 0.50), (0.707, 0.50))]:
        canvas.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16, linewidth=1.4, color="#7A8991"))

    canvas.text(
        0.5,
        0.085,
        "Tabular volume, comparability, and landscape support are distinct properties.",
        fontsize=9.5,
        fontweight="bold",
        color="#18323B",
        ha="center",
        va="center",
    )
    _save(fig, output_path)


def plot_spatial_representativeness(
    points: Sequence[dict[str, object]],
    geojson: dict[str, object],
    biome_rows: Sequence[dict[str, object]],
    state_rows: Sequence[dict[str, object]],
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(18, 14.5))
    grid = fig.add_gridspec(
        4,
        6,
        height_ratios=[1, 1, 0.07, 0.9],
        hspace=0.24,
        wspace=0.28,
    )
    map_axes = []
    all_values = np.asarray([float(point["carbon_median_g_kg"]) for point in points])
    log_values = np.log1p(all_values)
    vmin, vmax = np.quantile(log_values, [0.01, 0.99])
    norm = Normalize(vmin=float(vmin), vmax=float(vmax), clip=True)
    scatter = None
    for index, source in enumerate(SOURCE_ORDER):
        row_index = index // 3
        column_index = (index % 3) * 2
        ax = fig.add_subplot(grid[row_index, column_index : column_index + 2])
        map_axes.append(ax)
        _plot_geojson_boundaries(ax, geojson)
        source_points = [point for point in points if point["source"] == source]
        scatter = ax.scatter(
            [float(point["longitude"]) for point in source_points],
            [float(point["latitude"]) for point in source_points],
            c=np.log1p([float(point["carbon_median_g_kg"]) for point in source_points]),
            cmap="viridis",
            norm=norm,
            s=3.2,
            alpha=0.62,
            linewidths=0,
            zorder=2,
        )
        ax.set_xlim(-74.5, -33.5)
        ax.set_ylim(-34.5, 6.0)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"{chr(97 + index)}) {SOURCE_LABELS[source]} (n={len(source_points):,})", loc="left", fontsize=10, fontweight="bold")

    colorbar_ax = fig.add_subplot(grid[2, 1:5])
    if scatter is not None:
        colorbar = fig.colorbar(scatter, cax=colorbar_ax, orientation="horizontal")
        ticks = colorbar.get_ticks()
        colorbar.set_ticks(ticks)
        colorbar.set_ticklabels([f"{math.expm1(tick):.1f}" for tick in ticks])
        colorbar.set_label("Median carbon concentration at each source-rounded-key pair (g kg$^{-1}$); common log1p scale")

    ax = fig.add_subplot(grid[3, 0:3])
    positions = np.arange(len(biome_rows))
    area_share = np.asarray([float(row["area_share_pct"]) for row in biome_rows])
    point_share = np.asarray([float(row["point_share_pct"]) for row in biome_rows])
    width = 0.38
    ax.barh(positions - width / 2, area_share, height=width, color="#AAB4B8", label="Land area")
    ax.barh(positions + width / 2, point_share, height=width, color="#3568A8", label="Rounded location keys")
    ax.set_yticks(positions, [str(row["biome_label_en"]) for row in biome_rows])
    ax.invert_yaxis()
    ax.set_xlabel("National share (%)")
    ax.set_title("g) Biome sampling share vs land area", loc="left", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", visible=False)

    ax = fig.add_subplot(grid[3, 3:6])
    finite = [row for row in state_rows if math.isfinite(float(row["log2_representation_ratio"]))]
    ordered = sorted(finite, key=lambda row: float(row["log2_representation_ratio"]))
    selected = ordered[:5] + ordered[-5:]
    values = np.asarray([float(row["log2_representation_ratio"]) for row in selected])
    colors = ["#C58B2A" if value < 0 else "#3568A8" for value in values]
    y = np.arange(len(selected))
    ax.barh(y, values, color=colors, height=0.68)
    ax.axvline(0, color="#4F5B62", linewidth=1)
    ax.set_yticks(y, [str(row["state"]) for row in selected])
    ax.set_xlabel("log2(rounded-key share / area share)")
    ax.set_title("h) Lowest and highest state sampling intensity", loc="left", fontweight="bold")
    ax.grid(axis="y", visible=False)

    fig.suptitle("Spatial sampling support and relative intensity", x=0.04, y=0.99, ha="left", fontsize=17, fontweight="bold")
    fig.text(0.04, 0.012, "The descriptive index compares the share of rounded location keys with land-area share. It does not establish probability-sample or environmental representativeness; positive values in panel h indicate greater intensity relative to area.", fontsize=8.5, color="#59666F")
    fig.subplots_adjust(left=0.055, right=0.985, top=0.95, bottom=0.07)
    _save(fig, output_path)


def plot_consolidated_spatial_support(
    points: Sequence[dict[str, object]],
    geojson: dict[str, object],
    output_path: Path,
) -> None:
    locations = aggregate_locations_across_sources(points)
    all_source_values = np.asarray(
        [float(point["carbon_median_g_kg"]) for point in points],
        dtype=float,
    )
    log_source_values = np.log1p(all_source_values)
    vmin, vmax = np.quantile(log_source_values, [0.01, 0.99])
    concentration_norm = Normalize(
        vmin=float(vmin),
        vmax=float(vmax),
        clip=True,
    )

    ordered_concentration = sorted(
        locations,
        key=lambda location: float(location["carbon_median_g_kg"]),
    )
    ordered_sources = sorted(
        locations,
        key=lambda location: int(location["source_count"]),
    )
    max_sources = max(int(location["source_count"]) for location in locations)
    source_norm = Normalize(vmin=0.5, vmax=max_sources + 0.5)

    fig = plt.figure(figsize=(14, 7.2))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[1, 0.055],
        hspace=0.12,
        wspace=0.12,
    )

    ax_a = fig.add_subplot(grid[0, 0])
    _plot_geojson_boundaries(ax_a, geojson)
    scatter_a = ax_a.scatter(
        [float(location["longitude"]) for location in ordered_concentration],
        [float(location["latitude"]) for location in ordered_concentration],
        c=np.log1p(
            [
                float(location["carbon_median_g_kg"])
                for location in ordered_concentration
            ]
        ),
        cmap="viridis",
        norm=concentration_norm,
        s=3.0,
        alpha=0.62,
        linewidths=0,
        rasterized=True,
        zorder=2,
    )
    ax_a.set_title(
        f"a) Median carbon concentration (n={len(locations):,} locations)",
        loc="left",
        fontweight="bold",
    )

    ax_b = fig.add_subplot(grid[0, 1])
    _plot_geojson_boundaries(ax_b, geojson)
    counts = np.asarray(
        [int(location["source_count"]) for location in ordered_sources],
        dtype=float,
    )
    scatter_b = ax_b.scatter(
        [float(location["longitude"]) for location in ordered_sources],
        [float(location["latitude"]) for location in ordered_sources],
        c=counts,
        cmap="cividis",
        norm=source_norm,
        s=2.7 + 2.0 * counts,
        alpha=0.72,
        linewidths=0,
        rasterized=True,
        zorder=2,
    )
    ax_b.set_title(
        "b) Number of contributing sources at each location",
        loc="left",
        fontweight="bold",
    )

    for ax in (ax_a, ax_b):
        ax.set_xlim(-74.5, -33.5)
        ax.set_ylim(-34.5, 6.0)
        ax.set_aspect("equal")
        ax.axis("off")

    colorbar_a = fig.colorbar(
        scatter_a,
        cax=fig.add_subplot(grid[1, 0]),
        orientation="horizontal",
    )
    ticks = colorbar_a.get_ticks()
    colorbar_a.set_ticks(ticks)
    colorbar_a.set_ticklabels([f"{math.expm1(tick):.1f}" for tick in ticks])
    colorbar_a.set_label(
        "Median carbon concentration (g kg$^{-1}$); common Figure 2 log1p scale"
    )

    colorbar_b = fig.colorbar(
        scatter_b,
        cax=fig.add_subplot(grid[1, 1]),
        orientation="horizontal",
        ticks=np.arange(1, max_sources + 1),
    )
    colorbar_b.set_label("Contributing sources")

    fig.suptitle(
        "Supplementary Figure S4. Consolidated spatial support across sources",
        x=0.04,
        y=0.99,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.04,
        0.012,
        "Each source contributes one median per rounded location. Shared coordinates do not establish identical or independent sampling events; no spatial interpolation was applied.",
        fontsize=8.5,
        color="#59666F",
    )
    fig.subplots_adjust(left=0.04, right=0.985, top=0.91, bottom=0.10)
    _save(fig, output_path)


def plot_source_distributions(
    points: Sequence[dict[str, object]],
    matched_rows: Sequence[dict[str, object]],
    matched_year_rows: Sequence[dict[str, object]],
    output_path: Path,
) -> None:
    values = {
        source: np.asarray(
            [float(point["carbon_median_g_kg"]) for point in points if point["source"] == source],
            dtype=float,
        )
        for source in SOURCE_ORDER
    }
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    ax = axes[0, 0]
    box = ax.boxplot(
        [values[source] for source in SOURCE_ORDER],
        tick_labels=[SOURCE_LABELS[source] for source in SOURCE_ORDER],
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#172126", "linewidth": 1.4},
    )
    for patch, source in zip(box["boxes"], SOURCE_ORDER):
        patch.set_facecolor(SOURCE_COLORS[source])
        patch.set_alpha(0.82)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylabel("Carbon concentration (g kg$^{-1}$), symlog scale")
    ax.tick_params(axis="x", rotation=28)
    ax.set_title("a) Distribution at source-coordinate level", loc="left", fontweight="bold")
    ax.grid(axis="x", visible=False)

    ax = axes[0, 1]
    for source in SOURCE_ORDER:
        sorted_values = np.sort(values[source])
        probability = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
        ax.plot(sorted_values, probability, color=SOURCE_COLORS[source], linewidth=2, label=SOURCE_LABELS[source])
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(left=0)
    ax.set_xlabel("Carbon concentration (g kg$^{-1}$), symlog scale")
    ax.set_ylabel("Empirical cumulative probability")
    ax.set_title("b) Complete distribution and upper tails", loc="left", fontweight="bold")
    ax.legend(fontsize=8, ncol=2)

    ordered = sorted(matched_rows, key=lambda row: int(row["matched_blocks"]), reverse=True)
    strict_lookup = {
        (row["source_a"], row["source_b"]): row for row in matched_year_rows
    }
    pair_labels = [str(row["pair_label"]) for row in ordered]
    positions = np.arange(len(ordered))

    ax = axes[1, 0]
    medians = np.asarray([float(row["median_difference_a_minus_b_g_kg"]) for row in ordered])
    q1 = np.asarray([float(row["q1_difference_g_kg"]) for row in ordered])
    q3 = np.asarray([float(row["q3_difference_g_kg"]) for row in ordered])
    ax.errorbar(
        medians,
        positions,
        xerr=np.vstack([medians - q1, q3 - medians]),
        fmt="o",
        color="#2F6B9A",
        ecolor="#2F6B9A",
        capsize=3,
        linewidth=2,
        label="Same coordinate + exact depth",
    )
    strict_x, strict_y = [], []
    for position, row in enumerate(ordered):
        strict = strict_lookup.get((row["source_a"], row["source_b"]))
        if strict:
            strict_x.append(float(strict["median_difference_a_minus_b_g_kg"]))
            strict_y.append(position)
        adjusted_p = row["wilcoxon_p_holm"]
        if adjusted_p is None:
            p_text = "test not run"
        else:
            p_value = float(adjusted_p)
            p_text = "p$_H$<0.001" if p_value < 0.001 else f"p$_H$={p_value:.3f}"
        ax.text(
            q3[position] if q3[position] >= medians[position] else medians[position],
            position + 0.22,
            f"n={int(row['matched_blocks']):,}; {p_text}",
            fontsize=7,
            color="#52616A",
            ha="center",
        )
    if strict_x:
        ax.scatter(strict_x, strict_y, marker="D", s=34, facecolor="white", edgecolor="#C58B2A", linewidth=1.5, label="Also same sample year")
    ax.axvline(0, color="#4F5B62", linewidth=1, linestyle="--")
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_yticks(positions, pair_labels)
    ax.invert_yaxis()
    ax.set_xlabel("Paired difference: source A - source B (g kg$^{-1}$)")
    ax.set_title("c) Same-point, exact-depth paired differences", loc="left", fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", visible=False)

    ax = axes[1, 1]
    near = np.asarray([float(row["near_identical_0_01_g_kg_share"]) for row in ordered])
    within = np.asarray([float(row["within_20_percent_share"]) for row in ordered])
    bar_height = 0.32
    ax.barh(positions - bar_height / 2, 100 * near, height=bar_height, color="#087F5B", label="Difference <= 0.01 g kg$^{-1}$")
    ax.barh(positions + bar_height / 2, 100 * within, height=bar_height, color="#3568A8", label="Relative difference <= 20%")
    strict_near_x, strict_near_y = [], []
    for position, row in enumerate(ordered):
        strict = strict_lookup.get((row["source_a"], row["source_b"]))
        if strict:
            strict_near_x.append(100 * float(strict["near_identical_0_01_g_kg_share"]))
            strict_near_y.append(position - bar_height / 2)
    if strict_near_x:
        ax.scatter(strict_near_x, strict_near_y, marker="D", s=28, facecolor="#F7F8F8", edgecolor="#C58B2A", linewidth=1.3, label="Same-year near-identical share")
    ax.set_yticks(positions, pair_labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 103)
    ax.set_xlabel("Matched coordinate blocks (%)")
    ax.set_title("d) Agreement after depth matching", loc="left", fontweight="bold")
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
    ax.grid(axis="y", visible=False)

    fig.suptitle("Unpaired source context and matched-source agreement", x=0.04, y=0.99, ha="left", fontsize=17, fontweight="bold")
    fig.text(0.04, 0.955, "Panels a-b are descriptive only. Paired inference in panels c-d requires the same coordinate and exact standardized top and bottom depths; diamonds additionally require the same year.", fontsize=9.5, color="#52616A")
    fig.text(0.04, 0.012, "Wilcoxon tests use one median difference per coordinate (or coordinate-year) and Holm correction. Matching often identifies republished observations, so agreement does not establish independent laboratory replication.", fontsize=8.5, color="#59666F")
    fig.tight_layout(rect=(0.02, 0.05, 0.99, 0.94))
    _save(fig, output_path)


def plot_supplementary_completeness(
    rows: Sequence[HarmonizedRow],
    source_rows: Sequence[dict[str, object]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
    labels = [str(row["source_label"]) for row in source_rows]
    positions = np.arange(len(labels))

    ax = axes[0, 0]
    values = [float(row["sample_year_reported_pct"]) for row in source_rows]
    ax.barh(positions, values, color="#3568A8")
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Records with sample year (%)")
    ax.set_title("a) Temporal metadata completeness", loc="left", fontweight="bold")
    ax.grid(axis="y", visible=False)

    ax = axes[0, 1]
    values = [float(row["analysis_method_reported_pct"]) for row in source_rows]
    ax.barh(positions, values, color="#C58B2A")
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Records with an identified analytical method (%)")
    ax.set_title("b) Laboratory metadata completeness", loc="left", fontweight="bold")
    ax.grid(axis="y", visible=False)

    ax = axes[1, 0]
    origins = Counter(_target_origin(row.target_kind) for row in rows)
    origin_order = ["Measured carbon", "Reported stock", "Proxy or fraction"]
    origin_colors = ["#3568A8", "#4C6A64", "#C58B2A"]
    bars = ax.bar(origin_order, [origins[key] for key in origin_order], color=origin_colors)
    ax.set_yscale("log")
    ax.set_ylabel("Harmonized records, logarithmic scale")
    ax.set_title("c) Original target supplied by the source", loc="left", fontweight="bold")
    ax.tick_params(axis="x", rotation=18)
    ax.grid(axis="x", visible=False)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.08, f"{int(bar.get_height()):,}", ha="center", va="bottom", fontsize=8)

    ax = axes[1, 1]
    basis = Counter(_stock_basis(row.stock_method) for row in rows)
    basis_order = ["Reported stock", "Observed bulk density", "Estimated bulk density"]
    basis_colors = ["#4C6A64", "#3568A8", "#C58B2A"]
    bars = ax.bar(basis_order, [basis[key] for key in basis_order], color=basis_colors)
    ax.set_ylabel("Harmonized records")
    ax.set_title("d) Information basis for standardized stock", loc="left", fontweight="bold")
    ax.tick_params(axis="x", rotation=18)
    ax.grid(axis="x", visible=False)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(basis.values()) * 0.02, f"{int(bar.get_height()):,}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Supplementary Figure S1. Metadata completeness and derivation pathways", x=0.04, y=0.99, ha="left", fontsize=16, fontweight="bold")
    fig.text(0.04, 0.012, "An identified method excludes 'not reported' and pseudo-sample categories. Estimated bulk density includes state-depth, depth-only, carbon-bin, and related mean-reference pathways.", fontsize=8.5, color="#59666F")
    fig.tight_layout(rect=(0.02, 0.05, 0.99, 0.95))
    _save(fig, output_path)


def plot_method_significance_english(
    method_rows: Sequence[dict[str, str]],
    pairwise: Sequence[dict[str, str]],
    depth_rows: Sequence[dict[str, str]],
    summary: dict[str, object],
    output_path: Path,
) -> None:
    codes = [row["method_code"] for row in method_rows]
    index = {code: position for position, code in enumerate(codes)}
    size = len(codes)
    fig = plt.figure(figsize=(16, 10.5))
    grid = fig.add_gridspec(2, 2, hspace=0.36, wspace=0.26)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    y = np.arange(size)
    family_colors = {"dry_combustion": "#2F6B9A", "wet_oxidation": "#C58B2A", "other": "#6C757D"}
    for position, row in enumerate(method_rows):
        median = float(row["median_g_kg"])
        q1 = float(row["q1_g_kg"])
        q3 = float(row["q3_g_kg"])
        color = family_colors[row["method_family"]]
        ax_a.plot([q1, q3], [position, position], color=color, linewidth=5, solid_capstyle="round")
        ax_a.scatter(median, position, s=55, facecolor="white", edgecolor=color, linewidth=1.8, zorder=3)
    ax_a.set_yticks(y, [f"{row['method_code']}  n={row['point_year_observations']}" for row in method_rows])
    ax_a.invert_yaxis()
    ax_a.set_xscale("log")
    ax_a.set_xlabel("Carbon concentration (g kg$^{-1}$), logarithmic scale")
    ax_a.set_title("a) Median and interquartile range", loc="left", fontweight="bold")
    ax_a.grid(axis="y", visible=False)
    ax_a.spines[["top", "right", "left"]].set_visible(False)
    ax_a.tick_params(axis="y", length=0)
    ax_a.legend(handles=[Patch(facecolor="#2F6B9A", label="Dry combustion"), Patch(facecolor="#C58B2A", label="Wet oxidation")], loc="lower right")

    delta = np.zeros((size, size), dtype=float)
    p_matrix = np.full((size, size), np.nan)
    for row in pairwise:
        i = index[row["method_a_code"]]
        j = index[row["method_b_code"]]
        value = float(row["cliffs_delta_a_over_b"])
        p_value = float(row["mann_whitney_p_holm"])
        delta[i, j], delta[j, i] = value, -value
        p_matrix[i, j] = p_matrix[j, i] = p_value
    cmap = LinearSegmentedColormap.from_list("blue_orange", ["#2F6B9A", "#F7F8F8", "#C58B2A"])
    image = ax_b.imshow(delta, cmap=cmap, norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1))
    ax_b.set_xticks(range(size), codes, rotation=45, ha="right")
    ax_b.set_yticks(range(size), codes)
    ax_b.tick_params(length=0, labelsize=7)
    for i in range(size):
        for j in range(size):
            label = "-" if i == j else f"{delta[i, j]:+.2f}"
            ax_b.text(j, i, label, ha="center", va="center", fontsize=5.2, color="white" if abs(delta[i, j]) > 0.58 else "#172126")
    ax_b.grid(False)
    ax_b.set_title("b) Pairwise effect size", loc="left", fontweight="bold")
    cb = fig.colorbar(image, ax=ax_b, fraction=0.045, pad=0.03)
    cb.set_label("Cliff's delta: row method versus column method")

    significant = p_matrix < 0.05
    strength = np.minimum(-np.log10(np.clip(p_matrix, 1e-300, 1)), 20)
    masked = np.ma.masked_where(~significant, strength)
    p_cmap = matplotlib.colormaps["Blues"].copy()
    p_cmap.set_bad("#E8ECEE")
    p_image = ax_c.imshow(masked, cmap=p_cmap, vmin=1.301, vmax=20)
    ax_c.set_xticks(range(size), codes, rotation=45, ha="right")
    ax_c.set_yticks(range(size), codes)
    ax_c.tick_params(length=0, labelsize=7)
    for i in range(size):
        for j in range(size):
            if i == j:
                label = "-"
            elif significant[i, j]:
                label = "***" if p_matrix[i, j] < 0.001 else "**" if p_matrix[i, j] < 0.01 else "*"
            else:
                label = "ns"
            ax_c.text(j, i, label, ha="center", va="center", fontsize=5.8, color="white" if significant[i, j] and strength[i, j] > 8 else "#172126")
    ax_c.grid(False)
    sig_pairs = int(summary["significant_pairwise_holm_0_05"])
    ax_c.set_title(f"c) Pairwise significance: {sig_pairs}/{len(pairwise)} pairs", loc="left", fontweight="bold")
    cb = fig.colorbar(p_image, ax=ax_c, fraction=0.045, pad=0.03)
    cb.set_label("-log10(Holm-adjusted p), capped at 20")

    depths = [row["depth_band_cm"] for row in depth_rows]
    effects = np.asarray([float(row["epsilon_squared"]) for row in depth_rows])
    bars = ax_d.barh(np.arange(len(depths)), effects, color="#C58B2A", edgecolor="#815A1B", height=0.62)
    ax_d.set_yticks(np.arange(len(depths)), [f"{depth} cm" for depth in depths])
    ax_d.invert_yaxis()
    ax_d.axvline(0.14, color="#4F5B62", linestyle="--", linewidth=1)
    for bar, row in zip(bars, depth_rows):
        effect = float(row["epsilon_squared"])
        p_value = float(row["p_value"])
        p_text = f"p={p_value:.1e}" if p_value < 0.001 else f"p={p_value:.3f}"
        ax_d.text(effect + 0.012, bar.get_y() + bar.get_height() / 2, f"{effect:.3f} | {p_text}", va="center", fontsize=7.5)
    ax_d.set_xlim(0, 0.95)
    ax_d.set_xlabel("Kruskal-Wallis epsilon squared")
    ax_d.set_title("d) Sensitivity across depth groups", loc="left", fontweight="bold")
    ax_d.grid(axis="y", visible=False)
    global_point = summary["global_point_year"]
    ax_d.text(0, 1.09, f"Point-year global test: H={global_point['kruskal_wallis_h']:.1f}, epsilon$^2$={global_point['epsilon_squared']:.3f}, p<10$^{{-100}}$", transform=ax_d.transAxes, fontsize=8.5, fontweight="bold")

    fig.suptitle("Differences in carbon concentration among reported analytical methods", x=0.04, y=0.99, ha="left", fontsize=17, fontweight="bold")
    fig.text(0.04, 0.958, "Records with directly measured carbon were aggregated by coordinate, sample year, and method before inference.", fontsize=9.5, color="#52616A")
    fig.text(0.04, 0.015, "Method codes are defined in Supplementary Table S5. The groups are observational, not causal method effects; source, study, region, depth, and soil conditions remain confounded.", fontsize=8.5, color="#59666F")
    fig.subplots_adjust(left=0.06, right=0.98, top=0.91, bottom=0.10)
    _save(fig, output_path)


def plot_method_linear_sensitivity(
    method_rows: Sequence[dict[str, str]],
    output_path: Path,
) -> None:
    codes = [row["method_code"] for row in method_rows]
    y = np.arange(len(codes))
    family_colors = {
        "dry_combustion": "#2F6B9A",
        "wet_oxidation": "#C58B2A",
        "other": "#6C757D",
    }

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    for position, row in enumerate(method_rows):
        median = float(row["median_g_kg"])
        q1 = float(row["q1_g_kg"])
        q3 = float(row["q3_g_kg"])
        color = family_colors[row["method_family"]]
        ax.plot(
            [q1, q3],
            [position, position],
            color=color,
            linewidth=5,
            solid_capstyle="round",
        )
        ax.scatter(
            median,
            position,
            s=55,
            facecolor="white",
            edgecolor=color,
            linewidth=1.8,
            zorder=3,
        )

    ax.set_yticks(
        y,
        [
            f"{row['method_code']}  n={row['point_year_observations']}"
            for row in method_rows
        ],
    )
    ax.invert_yaxis()
    ax.set_xlim(left=0)
    ax.set_xlabel("Carbon concentration (g kg$^{-1}$), linear scale")
    ax.set_title(
        "Supplementary Figure S3. Analytical-method summaries on a linear scale",
        loc="left",
        fontweight="bold",
    )
    ax.grid(axis="y", visible=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(
        handles=[
            Patch(facecolor="#2F6B9A", label="Dry combustion"),
            Patch(facecolor="#C58B2A", label="Wet oxidation"),
        ],
        loc="lower right",
    )
    fig.text(
        0.08,
        0.025,
        "The medians, interquartile ranges, and point-year counts are identical to Figure 4a; only the display scale differs.",
        fontsize=8.5,
        color="#59666F",
    )
    fig.subplots_adjust(left=0.16, right=0.97, top=0.88, bottom=0.13)
    _save(fig, output_path)


def depth_harmonization_statistics(
    processed_dir: Path,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, list[float]],
]:
    grouped_rows = _read_csv(
        processed_dir / "soil_target_Brasil_Stock_Group_Depth.csv"
    )
    audit_rows = _read_csv(
        processed_dir / "soil_target_Brasil_Stock_Group_Depth_audit.csv"
    )

    relative_changes: dict[str, list[float]] = {}
    band_rows = []
    for label, stock_field, coverage_field in DEPTH_STOCK_BANDS:
        stock_values = [
            value
            for row in grouped_rows
            if (value := _float(row.get(stock_field))) is not None
        ]
        coverage = np.asarray(
            [_float(row.get(coverage_field)) or 0.0 for row in audit_rows],
            dtype=float,
        )
        changes = []
        for row in audit_rows:
            if row.get("DEPTH_HARMONIZATION_METHOD") != "equal_area_spline":
                continue
            estimate = _float(row.get(stock_field))
            reference = _float(row.get(f"OVERLAP_REFERENCE_{stock_field}"))
            if estimate is None or reference is None or abs(reference) <= 1e-8:
                continue
            changes.append((estimate - reference) / reference * 100)
        relative_changes[label] = changes
        band_rows.append(
            {
                "depth_band_cm": label,
                "assigned_targets": len(stock_values),
                "median_stock_Mg_ha": float(np.median(stock_values)),
                "stock_q1_Mg_ha": float(np.quantile(stock_values, 0.25)),
                "stock_q3_Mg_ha": float(np.quantile(stock_values, 0.75)),
                "fully_covered_profiles": int(np.sum(coverage >= 1 - 1e-8)),
                "partially_covered_profiles": int(
                    np.sum((coverage > 1e-8) & (coverage < 1 - 1e-8))
                ),
                "uncovered_profiles": int(np.sum(coverage <= 1e-8)),
                "spline_relative_change_median_pct": (
                    float(np.median(changes)) if changes else None
                ),
                "spline_relative_change_q1_pct": (
                    float(np.quantile(changes, 0.25)) if changes else None
                ),
                "spline_relative_change_q3_pct": (
                    float(np.quantile(changes, 0.75)) if changes else None
                ),
            }
        )

    source_rows = []
    for source in SOURCE_ORDER:
        rows = [row for row in audit_rows if row.get("SOURCE_ID") == source]
        source_rows.append(
            {
                "source_id": source,
                "source_label": SOURCE_LABELS[source],
                "audited_profiles": len(rows),
                "equal_area_spline_profiles": sum(
                    row.get("DEPTH_HARMONIZATION_METHOD") == "equal_area_spline"
                    for row in rows
                ),
                "overlap_allocation_profiles": sum(
                    row.get("DEPTH_HARMONIZATION_METHOD")
                    == "mass_preserving_overlap"
                    for row in rows
                ),
                "overlapping_input_profiles": sum(
                    (row.get("OVERLAPPING_INPUT_LAYERS") or "").lower() == "true"
                    for row in rows
                ),
            }
        )

    reasons = Counter(row.get("SPLINE_REASON") or "" for row in audit_rows)
    applied_pre_errors = [
        abs(value)
        for row in audit_rows
        if row.get("DEPTH_HARMONIZATION_METHOD") == "equal_area_spline"
        and (value := _float(row.get("SPLINE_PRE_RESCALE_MASS_BALANCE_ERROR_PCT")))
        is not None
    ]
    post_errors = [
        abs(_float(row.get("MASS_BALANCE_ERROR_PCT")) or 0.0) for row in audit_rows
    ]
    summary: dict[str, object] = {
        "grouped_profiles": len(grouped_rows),
        "audited_profiles": len(audit_rows),
        "assigned_targets": sum(
            _float(row.get(field)) is not None
            for row in grouped_rows
            for _label, field, _coverage in DEPTH_STOCK_BANDS
        ),
        "equal_area_spline_profiles": reasons["applied"],
        "overlap_allocation_profiles": len(audit_rows) - reasons["applied"],
        "mass_balance_guard_profiles": reasons["mass_balance_guard"],
        "zero_reference_stock_profiles": reasons["zero_reference_stock"],
        "overlapping_input_profiles": sum(
            (row.get("OVERLAPPING_INPUT_LAYERS") or "").lower() == "true"
            for row in audit_rows
        ),
        "max_post_harmonization_mass_error_pct": max(post_errors, default=0.0),
        "applied_spline_abs_pre_error_median_pct": float(
            np.median(applied_pre_errors)
        ),
        "applied_spline_abs_pre_error_p95_pct": float(
            np.quantile(applied_pre_errors, 0.95)
        ),
        "extrapolated_profiles": sum(
            (row.get("EXTRAPOLATED") or "").lower() == "true" for row in audit_rows
        ),
        "spline_reason_counts": dict(reasons),
    }
    return summary, band_rows, source_rows, relative_changes


def plot_depth_harmonization_audit(
    summary: dict[str, object],
    band_rows: Sequence[dict[str, object]],
    source_rows: Sequence[dict[str, object]],
    relative_changes: Mapping[str, list[float]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))
    labels = [str(row["depth_band_cm"]) for row in band_rows]
    positions = np.arange(len(labels))

    ax = axes[0, 0]
    targets = np.asarray([int(row["assigned_targets"]) for row in band_rows])
    medians = np.asarray([float(row["median_stock_Mg_ha"]) for row in band_rows])
    bars = ax.bar(positions, targets, width=0.62, color="#087F5B")
    ax.bar_label(bars, labels=[f"{value:,}" for value in targets], padding=3, fontsize=8)
    ax.set_xticks(positions, labels)
    ax.set_xlabel("Standard interval (cm)")
    ax.set_ylabel("Targets in standard intervals")
    ax.set_ylim(0, max(targets) * 1.18)
    median_axis = ax.twinx()
    median_axis.plot(positions, medians, color="#B43A4A", marker="o", linewidth=2)
    median_axis.set_ylabel("Median stock (Mg ha$^{-1}$)", color="#B43A4A")
    median_axis.tick_params(axis="y", colors="#B43A4A")
    ax.set_title("a) Target availability and median stock", loc="left", fontweight="bold")
    ax.grid(axis="x", visible=False)

    ax = axes[0, 1]
    y = np.arange(len(source_rows))
    spline = np.asarray([int(row["equal_area_spline_profiles"]) for row in source_rows])
    overlap = np.asarray([int(row["overlap_allocation_profiles"]) for row in source_rows])
    profile_total = spline + overlap
    spline_pct = 100 * spline / profile_total
    overlap_pct = 100 * overlap / profile_total
    ax.barh(y, spline_pct, color="#3568A8", label="Equal-area spline")
    ax.barh(y, overlap_pct, left=spline_pct, color="#C58B2A", label="Overlap allocation")
    ax.set_yticks(
        y,
        [
            f"{row['source_label']}  (n={int(row['audited_profiles']):,})"
            for row in source_rows
        ],
    )
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Audited operational profiles (%)")
    ax.set_title("b) Method selected by source", loc="left", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", visible=False)

    ax = axes[1, 0]
    full = np.asarray([int(row["fully_covered_profiles"]) for row in band_rows])
    partial = np.asarray([int(row["partially_covered_profiles"]) for row in band_rows])
    none = np.asarray([int(row["uncovered_profiles"]) for row in band_rows])
    total = full + partial + none
    ax.bar(positions, 100 * full / total, color="#087F5B", label="Full")
    ax.bar(
        positions,
        100 * partial / total,
        bottom=100 * full / total,
        color="#C58B2A",
        label="Partial",
    )
    ax.bar(
        positions,
        100 * none / total,
        bottom=100 * (full + partial) / total,
        color="#CBD2D6",
        label="None",
    )
    ax.set_xticks(positions, labels)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Standard interval (cm)")
    ax.set_ylabel("Audited profiles (%)")
    ax.set_title("c) Observed support; partial standard intervals remain empty", loc="left", fontweight="bold")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(axis="x", visible=False)

    ax = axes[1, 1]
    values = [relative_changes[label] for label in labels]
    box = ax.boxplot(values, tick_labels=labels, patch_artist=True, showfliers=False)
    for patch in box["boxes"]:
        patch.set_facecolor("#86AEC7")
        patch.set_alpha(0.9)
    ax.axhline(0, color="#4F5B62", linestyle="--", linewidth=1)
    ax.set_xlabel("Standard interval (cm)")
    ax.set_ylabel("Spline minus overlap reference (%)")
    ax.set_title("d) Spline redistributes mass within profiles", loc="left", fontweight="bold")
    ax.grid(axis="x", visible=False)

    fig.suptitle(
        "Supplementary Figure S2. Audit of standard-interval harmonization",
        x=0.04,
        y=0.99,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.04,
        0.012,
        f"Splines were accepted for {int(summary['equal_area_spline_profiles']):,} profiles and rejected when pre-rescale mass error exceeded 10%. All final estimates had zero aggregate mass-balance error across fully supported standard intervals; no extrapolation was used.",
        fontsize=8.5,
        color="#59666F",
    )
    fig.tight_layout(rect=(0.02, 0.05, 0.99, 0.95))
    _save(fig, output_path)


def build_manuscript_assets(
    harmonized_path: Path,
    processed_dir: Path,
    comparison_dir: Path,
    manuscript_dir: Path,
    state_geojson_path: Path,
    biome_shapefile_path: Path,
) -> dict[str, object]:
    _plot_style()
    figures_dir = manuscript_dir / "figures"
    tables_dir = manuscript_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    rows = load_harmonized_rows(harmonized_path)
    points = aggregate_source_points(rows)
    sources = source_summary(rows, points)
    flow = data_flow_summary(processed_dir)
    flow["exact_coordinate_pairs"] = len(
        {(row.latitude, row.longitude) for row in rows}
    )
    flow["rounded_location_keys"] = len(
        {(round(row.latitude, 5), round(row.longitude, 5)) for row in rows}
    )
    # Retained for backward compatibility with existing tables and summaries.
    flow["unique_coordinates"] = flow["rounded_location_keys"]
    flow["source_coordinate_pairs"] = len(points)
    overlaps = overlap_rows(comparison_dir / "sobreposicao_espacial_fontes.csv")
    matched = matched_source_comparison(rows, require_same_year=False)
    matched_year = matched_source_comparison(rows, require_same_year=True)
    geojson = json.loads(state_geojson_path.read_text(encoding="utf-8-sig"))
    biome_area = biome_areas(biome_shapefile_path)
    state_area = state_areas(geojson)
    biome_rows, state_rows = representation_statistics(rows, biome_area, state_area)
    depth_summary, depth_bands, depth_sources, depth_changes = (
        depth_harmonization_statistics(processed_dir)
    )

    _write_csv(tables_dir / "source_summary.csv", sources)
    _write_csv(tables_dir / "biome_representativeness.csv", biome_rows)
    _write_csv(tables_dir / "state_representativeness.csv", state_rows)
    _write_csv(tables_dir / "source_overlap.csv", overlaps)
    _write_csv(tables_dir / "source_matched_exact_depth.csv", matched)
    _write_csv(tables_dir / "source_matched_exact_depth_year.csv", matched_year)
    _write_csv(tables_dir / "data_flow.csv", [flow])
    _write_csv(tables_dir / "depth_harmonization_summary.csv", depth_bands)
    _write_csv(tables_dir / "depth_harmonization_by_source.csv", depth_sources)

    plot_workflow_provenance(flow, sources, overlaps, figures_dir / "figure_1_workflow_provenance.png")
    plot_spatial_representativeness(points, geojson, biome_rows, state_rows, figures_dir / "figure_2_spatial_representativeness.png")
    plot_consolidated_spatial_support(
        points,
        geojson,
        figures_dir / "figure_s4_consolidated_spatial_support.png",
    )
    plot_source_distributions(
        points,
        matched,
        matched_year,
        figures_dir / "figure_3_source_distributions.png",
    )
    method_summary = json.loads((comparison_dir / "significancia_metodos_resumo.json").read_text(encoding="utf-8"))
    plot_method_significance_english(
        _read_csv(comparison_dir / "significancia_metodos_resumo.csv"),
        _read_csv(comparison_dir / "significancia_metodos_pares.csv"),
        _read_csv(comparison_dir / "significancia_metodos_profundidade.csv"),
        method_summary,
        figures_dir / "figure_4_analytical_methods.png",
    )
    plot_method_linear_sensitivity(
        _read_csv(comparison_dir / "significancia_metodos_resumo.csv"),
        figures_dir / "figure_s3_analytical_methods_linear.png",
    )
    plot_supplementary_completeness(rows, sources, figures_dir / "figure_s1_metadata_completeness.png")
    plot_depth_harmonization_audit(
        depth_summary,
        depth_bands,
        depth_sources,
        depth_changes,
        figures_dir / "figure_s2_depth_harmonization.png",
    )
    plot_graphical_abstract(
        points,
        geojson,
        sources,
        flow,
        figures_dir / "graphical_abstract.png",
    )

    target_origins = Counter(_target_origin(row.target_kind) for row in rows)
    stock_bases = Counter(_stock_basis(row.stock_method) for row in rows)
    summary = {
        "harmonized_rows": len(rows),
        "sources": len(SOURCE_ORDER),
        "datasets": len({row.dataset for row in rows}),
        "exact_coordinate_pairs": int(flow["exact_coordinate_pairs"]),
        "rounded_location_keys": int(flow["rounded_location_keys"]),
        "unique_coordinates": int(flow["rounded_location_keys"]),
        "source_coordinate_pairs": len(points),
        "flow": flow,
        "target_origins": dict(target_origins),
        "stock_bases": dict(stock_bases),
        "source_matched_exact_depth": matched,
        "source_matched_exact_depth_year": matched_year,
        "method_significance": method_summary,
        "biome_representativeness": biome_rows,
        "depth_harmonization": depth_summary,
        "outputs": {
            "figures": [str(path.resolve()) for path in sorted(figures_dir.glob("*.png"))],
            "tables": [str(path.resolve()) for path in sorted(tables_dir.glob("*.csv"))],
        },
    }
    (manuscript_dir / "analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
