from __future__ import annotations

import csv
import html
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "carbono_solo_matplotlib"))

import folium
import matplotlib
import numpy as np
matplotlib.use("Agg")
from branca.element import Element, MacroElement
from folium import plugins
from jinja2 import Template
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from scipy import stats
from scipy.spatial.distance import jensenshannon

SOURCE_ORDER = [
    "mapbiomas_soc",
    "wosis_brazil",
    "bdsolos_embrapa",
    "soildata_dataset",
    "israd_brazil",
    "hybras_sgb",
]

SOURCE_LABELS = {
    "mapbiomas_soc": "MapBiomas Solo C3",
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

DEPTH_ORDER = ["0-5", "5-15", "15-30", "30-60", "60-100", ">100"]
BIOME_ORDER = [
    "Amazônia",
    "Cerrado",
    "Mata Atlântica",
    "Caatinga",
    "Pampa",
    "Pantanal",
]

STATE_CODES = {
    "11": ("RO", "Rondônia"), "12": ("AC", "Acre"), "13": ("AM", "Amazonas"),
    "14": ("RR", "Roraima"), "15": ("PA", "Pará"), "16": ("AP", "Amapá"),
    "17": ("TO", "Tocantins"), "21": ("MA", "Maranhão"), "22": ("PI", "Piauí"),
    "23": ("CE", "Ceará"), "24": ("RN", "Rio Grande do Norte"),
    "25": ("PB", "Paraíba"), "26": ("PE", "Pernambuco"), "27": ("AL", "Alagoas"),
    "28": ("SE", "Sergipe"), "29": ("BA", "Bahia"), "31": ("MG", "Minas Gerais"),
    "32": ("ES", "Espírito Santo"), "33": ("RJ", "Rio de Janeiro"),
    "35": ("SP", "São Paulo"), "41": ("PR", "Paraná"),
    "42": ("SC", "Santa Catarina"), "43": ("RS", "Rio Grande do Sul"),
    "50": ("MS", "Mato Grosso do Sul"), "51": ("MT", "Mato Grosso"),
    "52": ("GO", "Goiás"), "53": ("DF", "Distrito Federal"),
}


@dataclass(frozen=True)
class Observation:
    source: str
    dataset: str
    latitude: float
    longitude: float
    state: str
    biome: str
    depth_top: float | None
    depth_bottom: float | None
    carbon: float | None
    stock_method: str


@dataclass(frozen=True)
class PointSummary:
    source: str
    latitude: float
    longitude: float
    state: str
    biome: str
    median: float
    q1: float
    q3: float
    minimum: float
    maximum: float
    layers: int
    depth_top: float | None
    depth_bottom: float | None
    datasets: tuple[str, ...]


class ComparisonMapScript(MacroElement):
    def __init__(self, script: str) -> None:
        super().__init__()
        self._name = "ComparisonMapScript"
        self._template = Template(
            "{% macro script(this, kwargs) %}\n" + script + "\n{% endmacro %}"
        )


def _float(value: str | None) -> float | None:
    text = (value or "").strip().replace(",", ".")
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _fmt_number(value: object, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return ""
        if value == 0:
            return "0"
        if abs(float(value)) < 0.0001:
            return f"{float(value):.3e}"
        return f"{float(value):.{digits}f}".rstrip("0").rstrip(".")
    return str(value)


def _pt(value: object, digits: int = 2) -> str:
    text = _fmt_number(value, digits)
    if not text:
        return "N/D"
    if "e" in text.lower():
        return text
    parts = text.split(".")
    integer = parts[0]
    sign = ""
    if integer.startswith("-"):
        sign, integer = "-", integer[1:]
    # Rebuild grouping without relying on locale configuration.
    grouped = f"{int(integer):,}".replace(",", ".") if integer else "0"
    return sign + grouped + ("," + parts[1] if len(parts) > 1 else "")


def depth_bucket(depth_top: float | None, depth_bottom: float | None) -> str | None:
    if depth_top is None and depth_bottom is None:
        return None
    if depth_top is None:
        midpoint = depth_bottom
    elif depth_bottom is None:
        midpoint = depth_top
    else:
        midpoint = (depth_top + depth_bottom) / 2
    if midpoint is None or midpoint < 0:
        return None
    if midpoint < 5:
        return "0-5"
    if midpoint < 15:
        return "5-15"
    if midpoint < 30:
        return "15-30"
    if midpoint < 60:
        return "30-60"
    if midpoint < 100:
        return "60-100"
    return ">100"


def cliff_delta_from_u(u_statistic: float, n_a: int, n_b: int) -> float:
    return 2 * float(u_statistic) / (n_a * n_b) - 1


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    count = len(p_values)
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [1.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * float(p_values[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def _effect_label(delta: float) -> str:
    magnitude = abs(delta)
    if magnitude < 0.147:
        return "desprezível"
    if magnitude < 0.33:
        return "pequeno"
    if magnitude < 0.474:
        return "moderado"
    return "grande"


def load_observations(path: Path) -> list[Observation]:
    observations: list[Observation] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        required = {
            "SOURCE_ID",
            "DATASET_ID",
            "LATITUDE",
            "LONGITUDE",
            "STATE",
            "BIOME",
            "CARBON_CONTENT_G_KG",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required harmonized columns: {sorted(missing)}")
        for row in reader:
            latitude = _float(row.get("LATITUDE"))
            longitude = _float(row.get("LONGITUDE"))
            if latitude is None or longitude is None:
                continue
            observations.append(
                Observation(
                    source=(row.get("SOURCE_ID") or "").strip(),
                    dataset=(row.get("DATASET_ID") or "").strip(),
                    latitude=latitude,
                    longitude=longitude,
                    state=(row.get("STATE") or "").strip(),
                    biome=(row.get("BIOME") or "").strip(),
                    depth_top=_float(row.get("DEPTH_TOP_CM")),
                    depth_bottom=_float(row.get("DEPTH_BOTTOM_CM")),
                    carbon=_float(row.get("CARBON_CONTENT_G_KG")),
                    stock_method=(row.get("STOCK_CALCULATION_METHOD") or "").strip(),
                )
            )
    return observations


def aggregate_points(observations: Iterable[Observation]) -> list[PointSummary]:
    groups: dict[tuple[str, float, float], list[Observation]] = defaultdict(list)
    for observation in observations:
        if observation.carbon is not None:
            key = (observation.source, round(observation.latitude, 5), round(observation.longitude, 5))
            groups[key].append(observation)

    summaries: list[PointSummary] = []
    for (source, latitude, longitude), rows in groups.items():
        values = np.asarray([row.carbon for row in rows if row.carbon is not None], dtype=float)
        tops = [row.depth_top for row in rows if row.depth_top is not None]
        bottoms = [row.depth_bottom for row in rows if row.depth_bottom is not None]
        state = Counter(row.state for row in rows if row.state).most_common(1)
        biome = Counter(row.biome for row in rows if row.biome).most_common(1)
        summaries.append(
            PointSummary(
                source=source,
                latitude=latitude,
                longitude=longitude,
                state=state[0][0] if state else "",
                biome=biome[0][0] if biome else "",
                median=float(np.median(values)),
                q1=float(np.quantile(values, 0.25)),
                q3=float(np.quantile(values, 0.75)),
                minimum=float(np.min(values)),
                maximum=float(np.max(values)),
                layers=len(values),
                depth_top=min(tops) if tops else None,
                depth_bottom=max(bottoms) if bottoms else None,
                datasets=tuple(sorted({row.dataset for row in rows if row.dataset})),
            )
        )
    return summaries


def _distribution_stats(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    q01, q05, q25, q50, q75, q95, q99 = np.quantile(
        array, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    )
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
    median = float(q50)
    return {
        "mean": mean,
        "standard_deviation": std,
        "coefficient_variation": std / mean if mean else math.nan,
        "minimum": float(np.min(array)),
        "p01": float(q01),
        "p05": float(q05),
        "q1": float(q25),
        "median": median,
        "q3": float(q75),
        "p95": float(q95),
        "p99": float(q99),
        "maximum": float(np.max(array)),
        "iqr": float(q75 - q25),
        "mad": float(np.median(np.abs(array - median))),
        "skewness": float(stats.skew(array, bias=False)) if len(array) > 2 else math.nan,
        "zero_share": float(np.mean(array == 0)),
    }


def descriptive_statistics(
    observations: Sequence[Observation], point_summaries: Sequence[PointSummary]
) -> list[dict[str, object]]:
    point_counts = Counter(point.source for point in point_summaries)
    results: list[dict[str, object]] = []
    for source in SOURCE_ORDER:
        rows = [row for row in observations if row.source == source]
        values = [row.carbon for row in rows if row.carbon is not None]
        if not values:
            continue
        summary = _distribution_stats(values)
        q3 = summary["q3"]
        outlier_limit = q3 + 1.5 * summary["iqr"]
        results.append(
            {
                "source_id": source,
                "source_label": SOURCE_LABELS[source],
                "rows": len(rows),
                "valid_carbon_rows": len(values),
                "unique_points": point_counts[source],
                "datasets": len({row.dataset for row in rows if row.dataset}),
                "states": len({row.state for row in rows if row.state}),
                "biomes": len({row.biome for row in rows if row.biome}),
                **summary,
                "tukey_high_outlier_share": float(np.mean(np.asarray(values) > outlier_limit)),
                "back_calculated_rows": sum(
                    "back_calculated_carbon_content" in row.stock_method for row in rows
                ),
            }
        )
    return results


def point_statistics(point_summaries: Sequence[PointSummary]) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for source in SOURCE_ORDER:
        values = [point.median for point in point_summaries if point.source == source]
        if values:
            results.append(
                {
                    "source_id": source,
                    "source_label": SOURCE_LABELS[source],
                    "unique_points": len(values),
                    **_distribution_stats(values),
                }
            )
    return results


def pairwise_statistics(point_summaries: Sequence[PointSummary]) -> list[dict[str, object]]:
    values = {
        source: np.asarray(
            [point.median for point in point_summaries if point.source == source], dtype=float
        )
        for source in SOURCE_ORDER
    }
    rows: list[dict[str, object]] = []
    raw_p_values: list[float] = []
    for source_a, source_b in combinations(SOURCE_ORDER, 2):
        a = values[source_a]
        b = values[source_b]
        mann_whitney = stats.mannwhitneyu(a, b, alternative="two-sided", method="asymptotic")
        ks = stats.ks_2samp(a, b, alternative="two-sided", method="auto")
        delta = cliff_delta_from_u(float(mann_whitney.statistic), len(a), len(b))
        ceiling = max(float(np.quantile(a, 0.995)), float(np.quantile(b, 0.995)), 1.0)
        bins = np.linspace(0, math.log1p(ceiling), 81)
        hist_a, _ = np.histogram(np.clip(np.log1p(a), 0, bins[-1]), bins=bins)
        hist_b, _ = np.histogram(np.clip(np.log1p(b), 0, bins[-1]), bins=bins)
        js_distance = float(jensenshannon(hist_a + 1e-12, hist_b + 1e-12, base=2))
        raw_p_values.append(float(mann_whitney.pvalue))
        rows.append(
            {
                "source_a": source_a,
                "source_a_label": SOURCE_LABELS[source_a],
                "source_b": source_b,
                "source_b_label": SOURCE_LABELS[source_b],
                "points_a": len(a),
                "points_b": len(b),
                "median_a_g_kg": float(np.median(a)),
                "median_b_g_kg": float(np.median(b)),
                "median_difference_a_minus_b_g_kg": float(np.median(a) - np.median(b)),
                "median_ratio_a_over_b": float(np.median(a) / np.median(b))
                if np.median(b)
                else math.nan,
                "mann_whitney_u": float(mann_whitney.statistic),
                "mann_whitney_p": float(mann_whitney.pvalue),
                "cliffs_delta_a_over_b": delta,
                "effect_magnitude": _effect_label(delta),
                "ks_d": float(ks.statistic),
                "ks_p": float(ks.pvalue),
                "jensen_shannon_distance": js_distance,
            }
        )
    adjusted = holm_adjust(raw_p_values)
    for row, p_value in zip(rows, adjusted):
        row["mann_whitney_p_holm"] = p_value
        row["significant_holm_0_05"] = p_value < 0.05
    return rows


def spatial_overlap_statistics(point_summaries: Sequence[PointSummary]) -> list[dict[str, object]]:
    lookup: dict[str, dict[tuple[float, float], float]] = defaultdict(dict)
    for point in point_summaries:
        lookup[point.source][(point.latitude, point.longitude)] = point.median
    results: list[dict[str, object]] = []
    for source_a, source_b in combinations(SOURCE_ORDER, 2):
        coordinates = sorted(set(lookup[source_a]).intersection(lookup[source_b]))
        if coordinates:
            a = np.asarray([lookup[source_a][coordinate] for coordinate in coordinates])
            b = np.asarray([lookup[source_b][coordinate] for coordinate in coordinates])
            if len(coordinates) >= 3 and np.std(a) > 0 and np.std(b) > 0:
                rho, p_value = stats.spearmanr(a, b)
            else:
                rho, p_value = math.nan, math.nan
            difference = a - b
            relative_close = np.abs(difference) <= 0.2 * np.maximum.reduce(
                [np.abs(a), np.abs(b), np.ones_like(a)]
            )
            near_identical = np.abs(difference) <= 0.01
            results.append(
                {
                    "source_a": source_a,
                    "source_a_label": SOURCE_LABELS[source_a],
                    "source_b": source_b,
                    "source_b_label": SOURCE_LABELS[source_b],
                    "shared_coordinates_5dp": len(coordinates),
                    "median_a_shared_g_kg": float(np.median(a)),
                    "median_b_shared_g_kg": float(np.median(b)),
                    "median_paired_difference_a_minus_b_g_kg": float(np.median(difference)),
                    "median_absolute_difference_g_kg": float(np.median(np.abs(difference))),
                    "within_20_percent_share": float(np.mean(relative_close)),
                    "near_identical_0_01_g_kg_share": float(np.mean(near_identical)),
                    "spearman_rho": float(rho),
                    "spearman_p": float(p_value),
                }
            )
        else:
            results.append(
                {
                    "source_a": source_a,
                    "source_a_label": SOURCE_LABELS[source_a],
                    "source_b": source_b,
                    "source_b_label": SOURCE_LABELS[source_b],
                    "shared_coordinates_5dp": 0,
                    "median_a_shared_g_kg": math.nan,
                    "median_b_shared_g_kg": math.nan,
                    "median_paired_difference_a_minus_b_g_kg": math.nan,
                    "median_absolute_difference_g_kg": math.nan,
                    "within_20_percent_share": math.nan,
                    "near_identical_0_01_g_kg_share": math.nan,
                    "spearman_rho": math.nan,
                    "spearman_p": math.nan,
                }
            )
    return results


def grouped_statistics(
    observations: Sequence[Observation], dimension: str, order: Sequence[str]
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[Observation]] = defaultdict(list)
    for row in observations:
        if row.carbon is None:
            continue
        group = depth_bucket(row.depth_top, row.depth_bottom) if dimension == "depth" else row.biome
        if group:
            grouped[(row.source, group)].append(row)
    results: list[dict[str, object]] = []
    for source in SOURCE_ORDER:
        for group in order:
            rows = grouped.get((source, group), [])
            if not rows:
                continue
            values = [row.carbon for row in rows if row.carbon is not None]
            q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
            results.append(
                {
                    "source_id": source,
                    "source_label": SOURCE_LABELS[source],
                    dimension: group,
                    "rows": len(values),
                    "unique_points": len(
                        {(round(row.latitude, 5), round(row.longitude, 5)) for row in rows}
                    ),
                    "mean_g_kg": float(np.mean(values)),
                    "q1_g_kg": float(q1),
                    "median_g_kg": float(median),
                    "q3_g_kg": float(q3),
                }
            )
    return results


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter=";")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _fmt_number(value, 8) for key, value in row.items()})


def _plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "axes.edgecolor": "#A7B0B7",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "#E3E8EB",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "legend.frameon": False,
        }
    )


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_distributions(point_summaries: Sequence[PointSummary], path: Path) -> None:
    values = {
        source: np.asarray(
            [point.median for point in point_summaries if point.source == source], dtype=float
        )
        for source in SOURCE_ORDER
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1, 1.25]})
    box = axes[0].boxplot(
        [values[source] for source in SOURCE_ORDER],
        labels=[SOURCE_LABELS[source] for source in SOURCE_ORDER],
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#172126", "linewidth": 1.5},
        whiskerprops={"color": "#62717B"},
        capprops={"color": "#62717B"},
    )
    for patch, source in zip(box["boxes"], SOURCE_ORDER):
        patch.set_facecolor(SOURCE_COLORS[source])
        patch.set_alpha(0.8)
    axes[0].set_yscale("symlog", linthresh=1)
    axes[0].set_ylabel("Concentração de carbono (g/kg), escala log")
    axes[0].set_title("Mediana por coordenada")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].grid(axis="x", visible=False)

    for source in SOURCE_ORDER:
        sorted_values = np.sort(values[source])
        probability = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
        axes[1].plot(
            sorted_values,
            probability,
            label=SOURCE_LABELS[source],
            color=SOURCE_COLORS[source],
            linewidth=2,
        )
    axes[1].set_xscale("symlog", linthresh=1)
    axes[1].set_xlim(left=0)
    axes[1].set_xlabel("Concentração de carbono (g/kg), escala log")
    axes[1].set_ylabel("Proporção acumulada")
    axes[1].set_title("Distribuição acumulada empírica")
    axes[1].legend(loc="lower right", fontsize=9)
    fig.suptitle("Distribuição da concentração de carbono por fonte", fontsize=16, fontweight="bold")
    fig.text(
        0.5,
        0.005,
        "Unidade de análise: mediana das camadas em cada coordenada e fonte. Outliers ocultos apenas no boxplot.",
        ha="center",
        color="#59666F",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    _save_figure(fig, path)


def plot_median_iqr(point_stats: Sequence[dict[str, object]], path: Path) -> None:
    lookup = {str(row["source_id"]): row for row in point_stats}
    fig, ax = plt.subplots(figsize=(10, 5.8))
    positions = np.arange(len(SOURCE_ORDER))
    for position, source in zip(positions, SOURCE_ORDER):
        row = lookup[source]
        median = float(row["median"])
        q1 = float(row["q1"])
        q3 = float(row["q3"])
        ax.errorbar(
            median,
            position,
            xerr=[[median - q1], [q3 - median]],
            fmt="o",
            color=SOURCE_COLORS[source],
            ecolor=SOURCE_COLORS[source],
            elinewidth=5,
            capsize=0,
            markersize=8,
        )
        ax.text(q3 + 0.7, position, f"{median:.2f}", va="center", color="#2B353B", fontsize=9)
    ax.set_yticks(positions, [SOURCE_LABELS[source] for source in SOURCE_ORDER])
    ax.invert_yaxis()
    ax.set_xlabel("Concentração de carbono (g/kg)")
    ax.set_title("Mediana e intervalo interquartil por fonte", fontweight="bold")
    ax.grid(axis="y", visible=False)
    maximum_q3 = max(float(lookup[source]["q3"]) for source in SOURCE_ORDER)
    ax.set_xlim(0, maximum_q3 * 1.13)
    fig.text(
        0.5,
        0.01,
        "Ponto = mediana; barra = Q1-Q3. Valores agregados por coordenada dentro de cada fonte.",
        ha="center",
        color="#59666F",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save_figure(fig, path)


def plot_depth_profiles(depth_stats: Sequence[dict[str, object]], path: Path) -> None:
    lookup = {(str(row["source_id"]), str(row["depth"])): row for row in depth_stats}
    labels = DEPTH_ORDER[:-1]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12, 6.5))
    for source in SOURCE_ORDER:
        medians = []
        q1 = []
        q3 = []
        for label in labels:
            row = lookup.get((source, label))
            medians.append(float(row["median_g_kg"]) if row else np.nan)
            q1.append(float(row["q1_g_kg"]) if row else np.nan)
            q3.append(float(row["q3_g_kg"]) if row else np.nan)
        medians_array = np.asarray(medians)
        ax.plot(
            x,
            medians_array,
            marker="o",
            linewidth=2.2,
            color=SOURCE_COLORS[source],
            label=SOURCE_LABELS[source],
        )
        ax.fill_between(x, q1, q3, color=SOURCE_COLORS[source], alpha=0.08)
    ax.set_xticks(x, [f"{label} cm" for label in labels])
    ax.set_ylabel("Concentração mediana (g/kg)")
    ax.set_xlabel("Faixa do ponto médio da camada")
    ax.set_title("Perfis de concentração por profundidade", fontweight="bold")
    ax.legend(ncol=2, fontsize=9)
    ax.set_ylim(bottom=0)
    fig.text(
        0.5,
        0.01,
        "Linhas = medianas; faixas translúcidas = Q1-Q3. A classe é definida pelo ponto médio entre topo e base.",
        ha="center",
        color="#59666F",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save_figure(fig, path)


def _plot_pairwise_heatmap(
    pairwise: Sequence[dict[str, object]],
    value_field: str,
    title: str,
    colorbar_label: str,
    path: Path,
    signed: bool,
) -> None:
    size = len(SOURCE_ORDER)
    matrix = np.zeros((size, size), dtype=float)
    for row in pairwise:
        i = SOURCE_ORDER.index(str(row["source_a"]))
        j = SOURCE_ORDER.index(str(row["source_b"]))
        value = float(row[value_field])
        matrix[i, j] = value
        matrix[j, i] = -value if signed else value
    fig, ax = plt.subplots(figsize=(9, 7.5))
    if signed:
        image = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    else:
        image = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=max(float(np.max(matrix)), 0.01))
    labels = [SOURCE_LABELS[source] for source in SOURCE_ORDER]
    ax.set_xticks(range(size), labels, rotation=35, ha="right")
    ax.set_yticks(range(size), labels)
    for i in range(size):
        for j in range(size):
            if i == j:
                text_value = "-"
            else:
                text_value = f"{matrix[i, j]:.2f}"
            color = "white" if abs(matrix[i, j]) > (0.48 if signed else 0.35) else "#172126"
            ax.text(j, i, text_value, ha="center", va="center", color=color, fontsize=9)
    ax.grid(False)
    ax.set_title(title, fontweight="bold", pad=14)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label)
    fig.tight_layout()
    _save_figure(fig, path)


def plot_coverage(
    descriptive: Sequence[dict[str, object]], point_stats: Sequence[dict[str, object]], path: Path
) -> None:
    row_lookup = {str(row["source_id"]): row for row in descriptive}
    point_lookup = {str(row["source_id"]): row for row in point_stats}
    labels = [SOURCE_LABELS[source] for source in SOURCE_ORDER]
    rows = np.asarray([int(row_lookup[source]["valid_carbon_rows"]) for source in SOURCE_ORDER])
    points = np.asarray([int(point_lookup[source]["unique_points"]) for source in SOURCE_ORDER])
    x = np.arange(len(SOURCE_ORDER))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12, 6))
    bars_rows = ax.bar(x - width / 2, rows, width, label="Camadas", color="#4A7C89")
    bars_points = ax.bar(x + width / 2, points, width, label="Coordenadas", color="#D08557")
    ax.set_yscale("log")
    ax.set_xticks(x, labels, rotation=28, ha="right")
    ax.set_ylabel("Quantidade, escala log")
    ax.set_title("Cobertura amostral por fonte", fontweight="bold")
    ax.legend()
    ax.grid(axis="x", visible=False)
    for bars in (bars_rows, bars_points):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.08,
                f"{int(bar.get_height()):,}".replace(",", "."),
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
            )
    fig.tight_layout()
    _save_figure(fig, path)


def plot_biome_heatmap(biome_stats: Sequence[dict[str, object]], path: Path) -> None:
    matrix = np.full((len(SOURCE_ORDER), len(BIOME_ORDER)), np.nan)
    counts = np.zeros_like(matrix)
    for row in biome_stats:
        source = str(row["source_id"])
        biome = str(row["biome"])
        if source in SOURCE_ORDER and biome in BIOME_ORDER:
            i = SOURCE_ORDER.index(source)
            j = BIOME_ORDER.index(biome)
            matrix[i, j] = float(row["median_g_kg"])
            counts[i, j] = int(row["rows"])
    fig, ax = plt.subplots(figsize=(11, 6.5))
    masked = np.ma.masked_invalid(matrix)
    color_norm = Normalize(vmin=float(np.nanmin(matrix)), vmax=float(np.nanmax(matrix)))
    color_map = matplotlib.colormaps["viridis"]
    image = ax.imshow(masked, cmap=color_map, norm=color_norm, aspect="auto")
    ax.set_xticks(range(len(BIOME_ORDER)), BIOME_ORDER, rotation=30, ha="right")
    ax.set_yticks(range(len(SOURCE_ORDER)), [SOURCE_LABELS[source] for source in SOURCE_ORDER])
    for i in range(len(SOURCE_ORDER)):
        for j in range(len(BIOME_ORDER)):
            if np.isfinite(matrix[i, j]):
                red, green, blue, _ = color_map(color_norm(matrix[i, j]))
                luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                color = "#172126" if luminance > 0.57 else "white"
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.1f}\nn={int(counts[i, j]):,}".replace(",", "."),
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=8,
                )
    ax.grid(False)
    ax.set_title("Mediana da concentração por fonte e bioma", fontweight="bold", pad=14)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    colorbar.set_label("Mediana (g/kg)")
    fig.tight_layout()
    _save_figure(fig, path)


def _geometry_lines(geometry: dict[str, object]) -> list[np.ndarray]:
    lines: list[np.ndarray] = []
    for feature in geometry.get("features", []):
        shape = feature.get("geometry", {})
        coordinates = shape.get("coordinates", [])
        if shape.get("type") == "Polygon":
            polygons = [coordinates]
        elif shape.get("type") == "MultiPolygon":
            polygons = coordinates
        else:
            continue
        for polygon in polygons:
            if polygon:
                lines.append(np.asarray(polygon[0], dtype=float))
    return lines


def plot_source_maps(
    point_summaries: Sequence[PointSummary], geometry: dict[str, object], path: Path
) -> None:
    values = np.asarray([point.median for point in point_summaries], dtype=float)
    low = float(np.quantile(values, 0.01))
    high = float(np.quantile(values, 0.99))
    norm = Normalize(vmin=math.log1p(low), vmax=math.log1p(high))
    lines = _geometry_lines(geometry)
    fig, axes = plt.subplots(2, 3, figsize=(14, 12), sharex=True, sharey=True)
    scatter = None
    for ax, source in zip(axes.flat, SOURCE_ORDER):
        points = [point for point in point_summaries if point.source == source]
        for line in lines:
            ax.plot(line[:, 0], line[:, 1], color="#7A8A91", linewidth=0.45, zorder=1)
        scatter = ax.scatter(
            [point.longitude for point in points],
            [point.latitude for point in points],
            c=np.log1p([point.median for point in points]),
            cmap="viridis",
            norm=norm,
            s=6,
            alpha=0.66,
            linewidths=0,
            rasterized=True,
            zorder=2,
        )
        ax.set_title(f"{SOURCE_LABELS[source]}  |  n={len(points):,}".replace(",", "."), fontweight="bold")
        ax.set_xlim(-74.5, -33.5)
        ax.set_ylim(-34.5, 5.7)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(False)
        ax.set_facecolor("#F5F7F7")
    if scatter is not None:
        colorbar = fig.colorbar(scatter, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
        ticks_original = np.asarray([low, 5, 10, 20, 50, 100, high])
        ticks_original = np.unique(np.clip(ticks_original, low, high))
        colorbar.set_ticks(np.log1p(ticks_original))
        colorbar.set_ticklabels([f"{value:.1f}" for value in ticks_original])
        colorbar.set_label("Concentração mediana por coordenada (g/kg); escala limitada a P1-P99")
    fig.suptitle("Cobertura espacial e concentração de carbono por fonte", fontsize=16, fontweight="bold")
    fig.text(0.5, 0.04, "Longitude", ha="center")
    fig.text(0.02, 0.5, "Latitude", va="center", rotation="vertical")
    fig.subplots_adjust(left=0.06, right=0.91, top=0.92, bottom=0.07, wspace=0.08, hspace=0.13)
    _save_figure(fig, path)


MAP_CSS = r"""
<style>
html, body { width: 100%; height: 100%; margin: 0; color: #172126; font-family: Inter, system-ui, sans-serif; }
.folium-map { background: #e8eef0; }
.carbon-panel { position: absolute; z-index: 1000; top: 16px; left: 16px; width: 340px; box-sizing: border-box;
  padding: 15px; border: 1px solid rgba(35,52,64,.16); border-radius: 7px; background: rgba(255,255,255,.96);
  box-shadow: 0 10px 28px rgba(20,34,45,.18); backdrop-filter: blur(8px); }
.carbon-heading { display: flex; justify-content: space-between; align-items: baseline; gap: 10px; margin-bottom: 12px; }
.carbon-heading strong { font-size: 16px; } .carbon-heading span { color: #68757d; font-size: 11px; }
.carbon-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 9px; }
.carbon-field label { display: block; margin-bottom: 4px; color: #52616a; font-size: 10px; font-weight: 700; text-transform: uppercase; }
.carbon-field select { width: 100%; height: 36px; box-sizing: border-box; padding: 0 8px; border: 1px solid #ccd5da;
  border-radius: 5px; background: white; color: #172126; font: inherit; font-size: 12px; }
.carbon-stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 7px; margin-top: 12px; padding-top: 11px; border-top: 1px solid #e2e7ea; }
.carbon-stats span { display:block; color:#6b7880; font-size:9px; text-transform:uppercase; }
.carbon-stats strong { display:block; margin-top:3px; font-size:13px; }
.carbon-legend { margin-top: 12px; } .carbon-bar { height: 8px; border-radius: 4px; background: linear-gradient(90deg,#440154,#3b528b,#21918c,#5ec962,#fde725); }
.carbon-labels { display:flex; justify-content:space-between; margin-top:4px; color:#66747c; font-size:9px; }
.carbon-status { margin-top: 9px; color:#66747c; font-size:10px; line-height:1.35; }
.carbon-popup-title { font-size:14px; font-weight:700; margin-bottom:7px; }
.carbon-popup-meta { color:#5d6a72; font-size:11px; margin-bottom:8px; }
.carbon-popup-table { width:100%; border-collapse:collapse; font-size:11px; }
.carbon-popup-table td { border-top:1px solid #edf1f2; padding:4px 0; }
.carbon-popup-table td:last-child { text-align:right; font-weight:650; }
.leaflet-popup-content-wrapper { border-radius: 6px; }
@media (max-width: 650px) { .carbon-panel { top:9px; left:9px; width:calc(100% - 66px); padding:11px; }
  .carbon-grid { grid-template-columns:1fr; } .carbon-heading { margin-bottom:8px; } }
</style>
"""

MAP_PANEL = r"""
<div class="carbon-panel" role="region" aria-label="Filtros do mapa comparativo">
  <div class="carbon-heading"><strong>Concentração de carbono</strong><span>g/kg · sem clusters</span></div>
  <div class="carbon-grid">
    <div class="carbon-field"><label for="carbon-source">Fonte</label><select id="carbon-source"><option value="">Todas as fontes</option></select></div>
    <div class="carbon-field"><label for="carbon-state">Estado</label><select id="carbon-state"><option value="">Brasil inteiro</option></select></div>
    <div class="carbon-field"><label for="carbon-biome">Bioma</label><select id="carbon-biome"><option value="">Todos os biomas</option></select></div>
    <div class="carbon-field"><label for="carbon-range">Concentração</label><select id="carbon-range">
      <option value="all">Todos os valores</option><option value="low">Até a mediana</option>
      <option value="mid">Mediana a P95</option><option value="high">Acima do P95</option></select></div>
  </div>
  <div class="carbon-stats"><div><span>Pontos</span><strong id="carbon-count">-</strong></div>
    <div><span>Mediana</span><strong id="carbon-median">-</strong></div><div><span>P95</span><strong id="carbon-p95">-</strong></div></div>
  <div class="carbon-legend"><div class="carbon-bar"></div><div class="carbon-labels">
    <span id="carbon-p01">-</span><span id="carbon-p25">-</span><span id="carbon-p50">-</span><span id="carbon-p75">-</span><span id="carbon-p99">-</span></div></div>
  <div id="carbon-status" class="carbon-status">Preparando pontos...</div>
</div>
"""

MAP_JS = r"""
const comparisonMap = __MAP__;
const comparisonPointLayer = __POINT_LAYER__;
const comparisonStateLayer = __STATE_LAYER__;
const comparisonPoints = __POINTS__;
const comparisonSources = __SOURCES__;
const comparisonStates = __STATES__;
const comparisonBiomes = __BIOMES__;
const comparisonDatasets = __DATASETS__;
const globalBreaks = __BREAKS__;
const numberBR = new Intl.NumberFormat("pt-BR", {maximumFractionDigits: 2});
const sourceSelect = document.getElementById("carbon-source");
const stateSelect = document.getElementById("carbon-state");
const biomeSelect = document.getElementById("carbon-biome");
const rangeSelect = document.getElementById("carbon-range");
const statusBox = document.getElementById("carbon-status");
const renderer = L.canvas({padding: .55, tolerance: 5});
const palette = ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"];
let renderToken = 0;

comparisonSources.forEach((source, index) => { const option=document.createElement("option"); option.value=index; option.textContent=source.label; sourceSelect.appendChild(option); });
comparisonStates.forEach((state, index) => { const option=document.createElement("option"); option.value=index; option.textContent=state; stateSelect.appendChild(option); });
comparisonBiomes.forEach((biome, index) => { const option=document.createElement("option"); option.value=index; option.textContent=biome; biomeSelect.appendChild(option); });

function quantile(values, p) { if (!values.length) return 0; const i=(values.length-1)*p, lo=Math.floor(i), hi=Math.ceil(i); return lo===hi?values[lo]:values[lo]+(values[hi]-values[lo])*(i-lo); }
function color(value) { const lo=Math.log1p(globalBreaks[0]), hi=Math.log1p(globalBreaks[4]); const ratio=Math.max(0,Math.min(.999,(Math.log1p(value)-lo)/(hi-lo||1))); return palette[Math.floor(ratio*palette.length)]; }
function valueText(value) { return `${numberBR.format(value)} g/kg`; }
function depthText(top,bottom) { if(top==null&&bottom==null)return "N/D"; return `${top==null?"?":numberBR.format(top)}-${bottom==null?"?":numberBR.format(bottom)} cm`; }
function popup(row) { const datasets=comparisonDatasets[row[13]].join("<br>"); return `<div class="carbon-popup-title">${comparisonSources[row[2]].label}</div>`+
  `<div class="carbon-popup-meta">${comparisonStates[row[3]]} · ${comparisonBiomes[row[4]]}<br>${row[0].toFixed(5)}, ${row[1].toFixed(5)}</div>`+
  `<table class="carbon-popup-table"><tr><td>Mediana</td><td>${valueText(row[5])}</td></tr><tr><td>Q1-Q3</td><td>${numberBR.format(row[6])}-${numberBR.format(row[7])}</td></tr>`+
  `<tr><td>Mín.-máx.</td><td>${numberBR.format(row[8])}-${numberBR.format(row[9])}</td></tr><tr><td>Camadas</td><td>${row[10]}</td></tr>`+
  `<tr><td>Profundidade</td><td>${depthText(row[11],row[12])}</td></tr><tr><td>Dataset(s)</td><td>${datasets}</td></tr></table>`; }
function fitState(stateIndex) { if(stateIndex===""){comparisonMap.fitBounds([[-34.2,-74.2],[5.5,-34.0]],{padding:[12,12]});return;}
  comparisonStateLayer.eachLayer(layer=>{if(layer.feature.properties.sigla===comparisonStates[Number(stateIndex)]) comparisonMap.fitBounds(layer.getBounds(),{padding:[25,25],maxZoom:8});}); }
function updateStateStyle(stateIndex) { comparisonStateLayer.eachLayer(layer=>{const active=stateIndex!==""&&layer.feature.properties.sigla===comparisonStates[Number(stateIndex)];
  layer.setStyle({color:active?"#C4513C":"#687B84",weight:active?2.2:1,fillColor:active?"#F0C75E":"#91A3AA",fillOpacity:active?.18:.035});}); }
function applyFilters(focus) { const token=++renderToken; comparisonPointLayer.clearLayers(); const source=sourceSelect.value,state=stateSelect.value,biome=biomeSelect.value,range=rangeSelect.value;
  updateStateStyle(state); if(focus)fitState(state); statusBox.textContent="Atualizando pontos...";
  window.setTimeout(()=>{if(token!==renderToken)return; let filtered=comparisonPoints.filter(row=>(source===""||row[2]===Number(source))&&(state===""||row[3]===Number(state))&&(biome===""||row[4]===Number(biome)));
    let base=filtered.map(row=>row[5]).sort((a,b)=>a-b), median=quantile(base,.5), p95=quantile(base,.95);
    if(range==="low")filtered=filtered.filter(row=>row[5]<=median); else if(range==="mid")filtered=filtered.filter(row=>row[5]>median&&row[5]<=p95); else if(range==="high")filtered=filtered.filter(row=>row[5]>p95);
    const values=filtered.map(row=>row[5]).sort((a,b)=>a-b), localMedian=quantile(values,.5), localP95=quantile(values,.95);
    function finish(){ document.getElementById("carbon-count").textContent=filtered.length.toLocaleString("pt-BR"); document.getElementById("carbon-median").textContent=valueText(localMedian);
      document.getElementById("carbon-p95").textContent=valueText(localP95); ["p01","p25","p50","p75","p99"].forEach((id,i)=>document.getElementById(`carbon-${id}`).textContent=numberBR.format(globalBreaks[i]));
      statusBox.textContent="Cada círculo representa uma coordenada dentro da fonte. Cores usam a mesma escala nacional P1-P99."; }
    function batch(start){if(token!==renderToken)return; const end=Math.min(start+700,filtered.length); for(let i=start;i<end;i++){const row=filtered[i]; const marker=L.circleMarker([row[0],row[1]],{radius:4,color:"#fff",weight:.45,opacity:.8,fillColor:color(row[5]),fillOpacity:.76,renderer,bubblingMouseEvents:false});
      marker.bindTooltip(`<strong>${comparisonSources[row[2]].label}</strong><br>${comparisonStates[row[3]]} · ${comparisonBiomes[row[4]]}<br>${valueText(row[5])}`,{sticky:true,direction:"top",opacity:.95}); marker.bindPopup(popup(row),{maxWidth:350}); comparisonPointLayer.addLayer(marker);}
      statusBox.textContent=`Desenhando ${end.toLocaleString("pt-BR")} de ${filtered.length.toLocaleString("pt-BR")} pontos...`; if(end<filtered.length)requestAnimationFrame(()=>batch(end)); else finish();}
    batch(0); },0); }
[sourceSelect,biomeSelect,rangeSelect].forEach(select=>select.addEventListener("change",()=>applyFilters(false))); stateSelect.addEventListener("change",()=>applyFilters(true));
comparisonStateLayer.eachLayer(layer=>layer.on("click",()=>{const index=comparisonStates.indexOf(layer.feature.properties.sigla); if(index>=0){stateSelect.value=index;applyFilters(true);}}));
comparisonMap.whenReady(()=>applyFilters(false));
"""


def build_interactive_map(
    point_summaries: Sequence[PointSummary], geometry: dict[str, object], output_path: Path
) -> dict[str, int]:
    geometry = json.loads(json.dumps(geometry, ensure_ascii=False))
    for feature in geometry.get("features", []):
        properties = feature.setdefault("properties", {})
        code = str(properties.get("codarea", ""))
        sigla, name = STATE_CODES.get(code, (code, code))
        properties["sigla"] = sigla
        properties["nome"] = name

    sources = [{"id": source, "label": SOURCE_LABELS[source]} for source in SOURCE_ORDER]
    states = sorted({point.state for point in point_summaries if point.state})
    biomes = [biome for biome in BIOME_ORDER if any(point.biome == biome for point in point_summaries)]
    source_index = {source: index for index, source in enumerate(SOURCE_ORDER)}
    state_index = {state: index for index, state in enumerate(states)}
    biome_index = {biome: index for index, biome in enumerate(biomes)}
    dataset_sets = sorted({point.datasets for point in point_summaries})
    dataset_index = {datasets: index for index, datasets in enumerate(dataset_sets)}
    points = [
        [
            point.latitude,
            point.longitude,
            source_index[point.source],
            state_index[point.state],
            biome_index[point.biome],
            round(point.median, 3),
            round(point.q1, 3),
            round(point.q3, 3),
            round(point.minimum, 3),
            round(point.maximum, 3),
            point.layers,
            round(point.depth_top, 2) if point.depth_top is not None else None,
            round(point.depth_bottom, 2) if point.depth_bottom is not None else None,
            dataset_index[point.datasets],
        ]
        for point in point_summaries
        if point.source in source_index and point.state in state_index and point.biome in biome_index
    ]
    values = np.asarray([point.median for point in point_summaries], dtype=float)
    breaks = [float(value) for value in np.quantile(values, [0.01, 0.25, 0.5, 0.75, 0.99])]

    map_obj = folium.Map(
        location=[-14.2, -51.9], zoom_start=4, min_zoom=3, max_zoom=18,
        tiles=None, control_scale=True, prefer_canvas=True, zoom_control=False,
    )
    folium.TileLayer("CartoDB positron", name="Mapa claro", show=True).add_to(map_obj)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", show=False).add_to(map_obj)
    state_layer = folium.GeoJson(
        geometry,
        name="Limites estaduais",
        style_function=lambda _: {"color": "#687B84", "weight": 1, "fillColor": "#91A3AA", "fillOpacity": 0.035},
        highlight_function=lambda _: {"color": "#C4513C", "weight": 2, "fillOpacity": 0.12},
        tooltip=folium.GeoJsonTooltip(fields=["nome", "sigla"], aliases=["Estado", "UF"]),
    ).add_to(map_obj)
    point_layer = folium.FeatureGroup(name="Pontos de concentração", show=True).add_to(map_obj)
    plugins.Fullscreen(position="topright", title="Tela cheia", title_cancel="Sair da tela cheia").add_to(map_obj)
    plugins.MousePosition(position="bottomleft", separator=" | ", prefix="Lat/Lon", num_digits=4).add_to(map_obj)
    folium.LayerControl(position="bottomright", collapsed=True).add_to(map_obj)
    map_obj.fit_bounds([[-34.2, -74.2], [5.5, -34.0]], padding=(12, 12))
    root = map_obj.get_root()
    root.header.add_child(Element("<title>Comparação das fontes de carbono do solo</title>"))
    root.header.add_child(Element(MAP_CSS))
    root.html.add_child(Element(MAP_PANEL))
    compact = (",", ":")
    script = (
        MAP_JS.replace("__MAP__", map_obj.get_name())
        .replace("__POINT_LAYER__", point_layer.get_name())
        .replace("__STATE_LAYER__", state_layer.get_name())
        .replace("__POINTS__", json.dumps(points, separators=compact, ensure_ascii=False))
        .replace("__SOURCES__", json.dumps(sources, separators=compact, ensure_ascii=False))
        .replace("__STATES__", json.dumps(states, separators=compact, ensure_ascii=False))
        .replace("__BIOMES__", json.dumps(biomes, separators=compact, ensure_ascii=False))
        .replace("__DATASETS__", json.dumps(dataset_sets, separators=compact, ensure_ascii=False))
        .replace("__BREAKS__", json.dumps(breaks, separators=compact))
    )
    map_obj.add_child(ComparisonMapScript(script))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    map_obj.save(str(output_path))
    return {"plotted_points": len(points), "output_bytes": output_path.stat().st_size}


def _html_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    header_html = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in row) + "</tr>")
    return f"<div class='table-wrap'><table><thead><tr>{header_html}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def _p_text(value: float) -> str:
    if value < 0.001:
        return "< 0,001"
    return _pt(value, 3)


def build_report(
    output_path: Path,
    input_path: Path,
    observations: Sequence[Observation],
    point_summaries: Sequence[PointSummary],
    descriptive: Sequence[dict[str, object]],
    point_stats: Sequence[dict[str, object]],
    pairwise: Sequence[dict[str, object]],
    overlap: Sequence[dict[str, object]],
    depth_stats: Sequence[dict[str, object]],
    biome_stats: Sequence[dict[str, object]],
    kruskal_h: float,
    kruskal_p: float,
) -> None:
    source_stats = {str(row["source_id"]): row for row in descriptive}
    point_lookup = {str(row["source_id"]): row for row in point_stats}
    closest = sorted(pairwise, key=lambda row: (abs(float(row["cliffs_delta_a_over_b"])), float(row["jensen_shannon_distance"])))[:3]
    divergent = sorted(pairwise, key=lambda row: abs(float(row["cliffs_delta_a_over_b"])), reverse=True)[:3]
    shared = sorted(overlap, key=lambda row: int(row["shared_coordinates_5dp"]), reverse=True)
    shared = [row for row in shared if int(row["shared_coordinates_5dp"]) > 0]
    total_back = sum(int(row["back_calculated_rows"]) for row in descriptive)
    total_datasets = len({row.dataset for row in observations if row.dataset})
    global_values = np.asarray([row.carbon for row in observations if row.carbon is not None])

    source_table = _html_table(
        ["Fonte", "Camadas", "Coordenadas", "Mediana", "Q1-Q3", "Média", "P95", "Máximo", "Zeros"],
        [
            [
                SOURCE_LABELS[source],
                _pt(source_stats[source]["valid_carbon_rows"], 0),
                _pt(point_lookup[source]["unique_points"], 0),
                f"{_pt(point_lookup[source]['median'])} g/kg",
                f"{_pt(point_lookup[source]['q1'])}-{_pt(point_lookup[source]['q3'])}",
                f"{_pt(point_lookup[source]['mean'])} g/kg",
                f"{_pt(point_lookup[source]['p95'])} g/kg",
                f"{_pt(point_lookup[source]['maximum'])} g/kg",
                f"{100 * float(source_stats[source]['zero_share']):.1f}%".replace(".", ","),
            ]
            for source in SOURCE_ORDER
        ],
    )
    divergence_table = _html_table(
        ["Comparação", "Medianas (g/kg)", "Delta de Cliff", "Magnitude", "Distância JS", "p Holm"],
        [
            [
                f"{row['source_a_label']} × {row['source_b_label']}",
                f"{_pt(row['median_a_g_kg'])} × {_pt(row['median_b_g_kg'])}",
                _pt(row["cliffs_delta_a_over_b"], 3),
                row["effect_magnitude"],
                _pt(row["jensen_shannon_distance"], 3),
                _p_text(float(row["mann_whitney_p_holm"])),
            ]
            for row in divergent
        ],
    )
    coincidence_table = _html_table(
        ["Comparação", "Medianas (g/kg)", "Delta de Cliff", "KS D", "Distância JS", "p Holm"],
        [
            [
                f"{row['source_a_label']} × {row['source_b_label']}",
                f"{_pt(row['median_a_g_kg'])} × {_pt(row['median_b_g_kg'])}",
                _pt(row["cliffs_delta_a_over_b"], 3),
                _pt(row["ks_d"], 3),
                _pt(row["jensen_shannon_distance"], 3),
                _p_text(float(row["mann_whitney_p_holm"])),
            ]
            for row in closest
        ],
    )
    overlap_table = _html_table(
        ["Fontes", "Coordenadas comuns", "Diferença mediana A-B", "Diferença absoluta", "Até 20%", "Quase idênticos", "Spearman ρ"],
        [
            [
                f"{row['source_a_label']} × {row['source_b_label']}",
                _pt(row["shared_coordinates_5dp"], 0),
                f"{_pt(row['median_paired_difference_a_minus_b_g_kg'])} g/kg",
                f"{_pt(row['median_absolute_difference_g_kg'])} g/kg",
                f"{100 * float(row['within_20_percent_share']):.1f}%".replace(".", ","),
                f"{100 * float(row['near_identical_0_01_g_kg_share']):.1f}%".replace(".", ","),
                _pt(row["spearman_rho"], 3),
            ]
            for row in shared
        ],
    )
    depth_lookup = {(str(row["source_id"]), str(row["depth"])): row for row in depth_stats}
    depth_table = _html_table(
        ["Fonte", *[f"{depth} cm" for depth in DEPTH_ORDER[:-1]]],
        [
            [SOURCE_LABELS[source]]
            + [
                _pt(depth_lookup[(source, depth)]["median_g_kg"])
                if (source, depth) in depth_lookup
                else "N/D"
                for depth in DEPTH_ORDER[:-1]
            ]
            for source in SOURCE_ORDER
        ],
    )

    largest = divergent[0]
    most_similar = closest[0]
    closest_shape = min(pairwise, key=lambda row: float(row["jensen_shannon_distance"]))
    generated = datetime.now().astimezone().strftime("%d/%m/%Y %H:%M %Z")
    report = f"""<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Comparação das fontes de concentração de carbono</title>
<style>
:root{{--ink:#172126;--muted:#5f6d75;--line:#dce3e6;--soft:#f4f7f7;--accent:#087f5b;}}
*{{box-sizing:border-box}} body{{margin:0;color:var(--ink);font-family:Inter,system-ui,-apple-system,"Segoe UI",sans-serif;line-height:1.55;background:white}}
header{{border-bottom:1px solid var(--line);background:#f7f9f9}} .wrap{{width:min(1180px,calc(100% - 36px));margin:auto}}
header .wrap{{padding:36px 0 28px}} h1{{font-size:30px;line-height:1.15;margin:0 0 10px;letter-spacing:0}} h2{{font-size:21px;margin:40px 0 12px;letter-spacing:0}}
h3{{font-size:16px;margin:24px 0 8px;letter-spacing:0}} p{{max-width:900px}} .subtitle{{color:var(--muted);font-size:15px;max-width:850px}}
.meta{{display:flex;flex-wrap:wrap;gap:18px;margin-top:18px;color:var(--muted);font-size:12px}} main{{padding:22px 0 48px}}
.findings{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:20px 0 6px}} .finding{{border-left:4px solid var(--accent);padding:10px 14px;background:var(--soft)}}
.finding strong{{display:block;font-size:19px;margin-bottom:3px}} .finding span{{font-size:12px;color:var(--muted)}}
.note{{border:1px solid var(--line);border-left:4px solid #b07a12;padding:12px 14px;background:#fffdf6;margin:16px 0;max-width:960px}}
.figure{{margin:18px 0 34px}} .figure img{{display:block;width:100%;height:auto;border:1px solid var(--line)}} .caption{{margin-top:7px;color:var(--muted);font-size:12px;max-width:1000px}}
.table-wrap{{overflow-x:auto;border:1px solid var(--line);margin:12px 0 24px}} table{{border-collapse:collapse;width:100%;font-size:12px}} th,td{{padding:9px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}
th:first-child,td:first-child{{text-align:left}} th{{background:#eef3f3;font-weight:700}} tbody tr:last-child td{{border-bottom:0}}
.map-frame{{width:100%;height:720px;border:1px solid var(--line)}} ul{{max-width:960px;padding-left:20px}} li{{margin-bottom:7px}} code{{background:#eef2f3;padding:2px 4px;border-radius:3px}}
.downloads{{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0 24px}} .downloads a{{color:#075e46;border:1px solid #a7c9bd;padding:7px 10px;border-radius:4px;text-decoration:none;font-size:12px}}
footer{{border-top:1px solid var(--line);padding:20px 0 34px;color:var(--muted);font-size:12px}}
@media(max-width:760px){{.findings{{grid-template-columns:1fr}}h1{{font-size:25px}}.map-frame{{height:600px}}}}
</style></head><body>
<header><div class="wrap"><h1>Comparação das seis fontes de concentração de carbono no solo</h1>
<div class="subtitle">Análise do campo <code>CARBON_CONTENT_G_KG</code> na base harmonizada brasileira, com estatísticas robustas, controle descritivo por profundidade e bioma, comparações entre fontes e cobertura espacial.</div>
<div class="meta"><span>Gerado em {generated}</span><span>{_pt(len(observations),0)} camadas</span><span>{_pt(len(point_summaries),0)} pares fonte-coordenada</span><span>{total_datasets} datasets</span></div></div></header>
<main class="wrap">
<section><h2>Resultado principal</h2>
<div class="findings"><div class="finding"><strong>{_pt(point_lookup['soildata_dataset']['median'])} g/kg</strong><span>maior mediana por coordenada: SoilData</span></div>
<div class="finding"><strong>{_pt(point_lookup['bdsolos_embrapa']['median'])} g/kg</strong><span>menor mediana por coordenada: BD Solos / Embrapa</span></div>
<div class="finding"><strong>{_pt(global_values.max())} g/kg</strong><span>máximo observado; as caudas são muito assimétricas</span></div></div>
<p>As fontes não compartilham uma única distribuição de concentração (Kruskal-Wallis H = {_pt(kruskal_h,2)}, p {_p_text(kruskal_p)}). A maior separação aparece entre <strong>{largest['source_a_label']}</strong> e <strong>{largest['source_b_label']}</strong>, com delta de Cliff {_pt(largest['cliffs_delta_a_over_b'],3)} ({largest['effect_magnitude']}). A maior coincidência de posição é entre <strong>{most_similar['source_a_label']}</strong> e <strong>{most_similar['source_b_label']}</strong>, com delta {_pt(most_similar['cliffs_delta_a_over_b'],3)}. Já a forma mais parecida é a de <strong>{closest_shape['source_a_label']}</strong> e <strong>{closest_shape['source_b_label']}</strong>, com distância Jensen-Shannon {_pt(closest_shape['jensen_shannon_distance'],3)}.</p>
<div class="note"><strong>Interpretação correta:</strong> diferença entre fontes não prova viés laboratorial. Profundidade, bioma, localização, época, desenho amostral e método analítico variam entre bases. Por isso, os testes foram feitos na mediana por coordenada e são acompanhados por perfis de profundidade e cobertura espacial.</div></section>

<section><h2>Estatísticas descritivas</h2>{source_table}
<div class="figure"><img src="graficos/01_distribuicao_fontes.png" alt="Boxplots e curvas acumuladas por fonte"><div class="caption">A escala logarítmica preserva os zeros e torna visíveis tanto o centro quanto as caudas. ISRaD tem poucos pontos e uma cauda particularmente pesada; MapBiomas é a única fonte com proporção relevante de valores zero.</div></div>
<div class="figure"><img src="graficos/02_mediana_iqr.png" alt="Mediana e intervalo interquartil por fonte"><div class="caption">SoilData apresenta o centro mais alto e maior dispersão central. WoSIS e BD Solos concentram-se em níveis mais baixos; MapBiomas e HYBRAS têm medianas próximas.</div></div></section>

<section><h2>Divergências e coincidências</h2><h3>Maiores divergências por tamanho de efeito</h3>{divergence_table}
<h3>Maior coincidência de posição (delta próximo de zero)</h3>{coincidence_table}
<div class="figure"><img src="graficos/04_heatmap_cliffs_delta.png" alt="Heatmap de delta de Cliff"><div class="caption">Delta positivo indica maior probabilidade de um valor da fonte da linha exceder um valor da fonte da coluna. O tamanho de efeito é mais informativo do que o p-valor diante do grande desbalanceamento amostral.</div></div>
<div class="figure"><img src="graficos/05_heatmap_jensen_shannon.png" alt="Heatmap de distância Jensen-Shannon"><div class="caption">A distância Jensen-Shannon compara a forma completa das distribuições em log(1+x): zero indica distribuições idênticas; valores maiores indicam formas mais distintas.</div></div></section>

<section><h2>Coincidência espacial e dependência entre fontes</h2>{overlap_table if shared else '<p>Não há coordenadas compartilhadas entre fontes.</p>'}
<p>O pareamento usa latitude e longitude arredondadas a cinco casas decimais. Valores “quase idênticos” diferem no máximo 0,01 g/kg. WoSIS e SoilData compartilham 400 coordenadas com valores essencialmente reproduzidos, enquanto MapBiomas se sobrepõe amplamente ao BD Solos. Isso indica dependência entre fontes e impede interpretar as seis bases como replicações totalmente independentes. O pareamento ainda não garante mesma data, profundidade ou protocolo de laboratório.</p>
<div class="figure"><img src="graficos/07_mapas_fontes.png" alt="Mapas estáticos das seis fontes"><div class="caption">Todos os painéis usam a mesma escala de cor P1-P99. A distribuição geográfica desigual evidencia que parte das diferenças nacionais decorre da composição espacial.</div></div></section>

<section><h2>Profundidade e bioma</h2>{depth_table}
<div class="figure"><img src="graficos/03_perfis_profundidade.png" alt="Perfis por profundidade"><div class="caption">As concentrações tendem a diminuir com a profundidade, embora intensidade e regularidade variem por fonte. Camadas foram classificadas pelo ponto médio; a análise não redistribui matematicamente horizontes largos.</div></div>
<div class="figure"><img src="graficos/08_biomas_fontes.png" alt="Medianas por fonte e bioma"><div class="caption">A matriz mostra mediana e número de camadas. Células ausentes revelam cobertura incompleta, especialmente nas fontes menores.</div></div></section>

<section><h2>Cobertura e qualidade</h2><div class="figure"><img src="graficos/06_cobertura_amostral.png" alt="Quantidade de camadas e coordenadas por fonte"><div class="caption">MapBiomas, WoSIS e BD Solos dominam o volume. ISRaD e HYBRAS devem ser interpretados com intervalos de incerteza mais amplos.</div></div>
<ul><li>O campo de concentração está preenchido em {_pt(len(global_values),0)} de {_pt(len(observations),0)} linhas válidas para coordenadas.</li>
<li>{total_back} concentrações foram retrocalculadas a partir de estoque, densidade, espessura e fragmentos; todas pertencem ao fluxo SoilData e estão identificadas no método de cálculo.</li>
<li>MapBiomas contém {100*float(source_stats['mapbiomas_soc']['zero_share']):.1f}% de zeros. Esse padrão deve ser checado contra limites de detecção, codificação de ausentes e definição da variável original.</li>
<li>Há redundância cruzada detectável: coordenadas e concentrações quase idênticas aparecem em mais de uma fonte. Separações de treino e teste devem ser feitas por coordenada e grupo de origem, não por linha.</li>
<li>As distribuições são assimétricas (cauda à direita), portanto mediana, IQR, ECDF e tamanhos de efeito foram priorizados.</li></ul></section>

<section><h2>Mapa interativo</h2><p>O mapa não usa agrupamento em clusters. Cada círculo é uma coordenada agregada dentro da fonte; filtros permitem isolar fonte, estado, bioma e faixa de concentração.</p>
<iframe class="map-frame" src="mapa_comparativo_concentracao_carbono.html" title="Mapa comparativo interativo"></iframe></section>

<section><h2>Arquivos auditáveis</h2><div class="downloads">
<a href="estatisticas_descritivas_por_fonte.csv">Descritivas por fonte</a><a href="estatisticas_por_coordenada_fonte.csv">Resumo por coordenada</a>
<a href="comparacoes_pareadas_fontes.csv">Comparações pareadas</a><a href="sobreposicao_espacial_fontes.csv">Sobreposição espacial</a>
<a href="estatisticas_por_fonte_profundidade.csv">Profundidade</a><a href="estatisticas_por_fonte_bioma.csv">Biomas</a><a href="resumo_analise.json">Resumo JSON</a></div></section>

<section><h2>Metodologia e limites</h2><ul>
<li>Descritivas gerais usam todas as camadas. Testes entre fontes usam a mediana das camadas por coordenada e fonte para reduzir pseudorreplicação.</li>
<li>Teste global: Kruskal-Wallis. Comparações pareadas: Mann-Whitney com correção de Holm, delta de Cliff, Kolmogorov-Smirnov e distância Jensen-Shannon.</li>
<li>Os limiares de delta de Cliff usados são: desprezível &lt; 0,147; pequeno &lt; 0,33; moderado &lt; 0,474; grande acima disso.</li>
<li>Não houve padronização por método laboratorial, ano ou suporte espacial. Antes de treinar um modelo, recomenda-se incluir fonte, profundidade, bioma e método como controles, ou calibrar fontes em observações pareadas.</li>
<li>Arquivo de entrada: <code>{html.escape(str(input_path))}</code>. CSVs de saída usam separador ponto e vírgula e UTF-8 com BOM.</li></ul></section>
</main><footer><div class="wrap">Carbono Solo · comparação reproduzível das fontes harmonizadas brasileiras.</div></footer></body></html>"""
    output_path.write_text(report, encoding="utf-8")


def run_source_comparison(input_path: Path, geometry_path: Path, output_dir: Path) -> dict[str, object]:
    input_path = input_path.resolve()
    geometry_path = geometry_path.resolve()
    output_dir = output_dir.resolve()
    charts_dir = output_dir / "graficos"
    output_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)
    _plot_style()

    observations = load_observations(input_path)
    observed_sources = {row.source for row in observations}
    missing_sources = set(SOURCE_ORDER).difference(observed_sources)
    if missing_sources:
        raise ValueError(f"Expected six sources; missing: {sorted(missing_sources)}")
    point_summaries = aggregate_points(observations)
    descriptive = descriptive_statistics(observations, point_summaries)
    points_stats = point_statistics(point_summaries)
    pairwise = pairwise_statistics(point_summaries)
    overlap = spatial_overlap_statistics(point_summaries)
    depth_stats = grouped_statistics(observations, "depth", DEPTH_ORDER)
    biome_stats = grouped_statistics(observations, "biome", BIOME_ORDER)

    point_arrays = [
        np.asarray([point.median for point in point_summaries if point.source == source])
        for source in SOURCE_ORDER
    ]
    kruskal = stats.kruskal(*point_arrays)

    _write_csv(output_dir / "estatisticas_descritivas_por_fonte.csv", descriptive)
    _write_csv(output_dir / "estatisticas_por_coordenada_fonte.csv", points_stats)
    _write_csv(output_dir / "comparacoes_pareadas_fontes.csv", pairwise)
    _write_csv(output_dir / "sobreposicao_espacial_fontes.csv", overlap)
    _write_csv(output_dir / "estatisticas_por_fonte_profundidade.csv", depth_stats)
    _write_csv(output_dir / "estatisticas_por_fonte_bioma.csv", biome_stats)

    plot_distributions(point_summaries, charts_dir / "01_distribuicao_fontes.png")
    plot_median_iqr(points_stats, charts_dir / "02_mediana_iqr.png")
    plot_depth_profiles(depth_stats, charts_dir / "03_perfis_profundidade.png")
    _plot_pairwise_heatmap(
        pairwise, "cliffs_delta_a_over_b", "Diferença probabilística entre fontes",
        "Delta de Cliff (linha versus coluna)", charts_dir / "04_heatmap_cliffs_delta.png", True,
    )
    _plot_pairwise_heatmap(
        pairwise, "jensen_shannon_distance", "Distância entre as formas das distribuições",
        "Distância Jensen-Shannon", charts_dir / "05_heatmap_jensen_shannon.png", False,
    )
    plot_coverage(descriptive, points_stats, charts_dir / "06_cobertura_amostral.png")
    geometry = json.loads(geometry_path.read_text(encoding="utf-8-sig"))
    plot_source_maps(point_summaries, geometry, charts_dir / "07_mapas_fontes.png")
    plot_biome_heatmap(biome_stats, charts_dir / "08_biomas_fontes.png")
    map_summary = build_interactive_map(
        point_summaries, geometry, output_dir / "mapa_comparativo_concentracao_carbono.html"
    )

    summary = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "input_file": str(input_path),
        "output_directory": str(output_dir),
        "sources": len(observed_sources),
        "datasets": len({row.dataset for row in observations if row.dataset}),
        "rows": len(observations),
        "rows_with_carbon": sum(row.carbon is not None for row in observations),
        "source_coordinate_pairs": len(point_summaries),
        "unique_coordinates_all_sources": len({(point.latitude, point.longitude) for point in point_summaries}),
        "back_calculated_carbon_rows": sum(
            row.carbon is not None and "back_calculated_carbon_content" in row.stock_method
            for row in observations
        ),
        "kruskal_wallis": {"h": float(kruskal.statistic), "p": float(kruskal.pvalue)},
        "map": map_summary,
        "source_counts": {
            source: {
                "rows": sum(row.source == source for row in observations),
                "points": sum(point.source == source for point in point_summaries),
            }
            for source in SOURCE_ORDER
        },
    }
    (output_dir / "resumo_analise.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    build_report(
        output_dir / "relatorio_comparativo_fontes_carbono.html",
        input_path,
        observations,
        point_summaries,
        descriptive,
        points_stats,
        pairwise,
        overlap,
        depth_stats,
        biome_stats,
        float(kruskal.statistic),
        float(kruskal.pvalue),
    )
    return summary
