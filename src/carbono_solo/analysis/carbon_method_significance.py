from __future__ import annotations

import csv
import json
import math
import os
import tempfile
import textwrap
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "carbono_solo_matplotlib"),
)

import matplotlib
import numpy as np
from scipy import stats

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Patch

from carbono_solo.collector.lab_methods import NOT_APPLICABLE_PSEUDO, NOT_REPORTED


DIRECT_TARGET_KINDS = {"carbon_content", "total_carbon_content"}
DEPTH_ORDER = ["0-5", "5-15", "15-30", "30-60", "60-100", ">100"]
FAMILY_COLORS = {
    "dry_combustion": "#2F6B9A",
    "wet_oxidation": "#C58B2A",
    "other": "#6C757D",
}
FAMILY_LABELS = {
    "dry_combustion": "Combustão seca",
    "wet_oxidation": "Oxidação úmida",
    "other": "Outro método",
}


@dataclass(frozen=True)
class MethodObservation:
    source_id: str
    dataset_id: str
    method: str
    latitude: float
    longitude: float
    sample_year: str
    depth_top_cm: float | None
    depth_bottom_cm: float | None
    carbon_g_kg: float


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


def load_direct_method_observations(path: Path) -> list[MethodObservation]:
    observations: list[MethodObservation] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        required = {
            "SOURCE_ID",
            "DATASET_ID",
            "LATITUDE",
            "LONGITUDE",
            "SAMPLE_YEAR",
            "DEPTH_TOP_CM",
            "DEPTH_BOTTOM_CM",
            "CARBON_CONTENT_G_KG",
            "ORIGINAL_TARGET_KIND",
            "CARBON_ANALYSIS_METHOD",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required harmonized columns: {sorted(missing)}")
        for row in reader:
            method = (row.get("CARBON_ANALYSIS_METHOD") or "").strip()
            normalized_method = _normalize(method)
            target_kind = (row.get("ORIGINAL_TARGET_KIND") or "").strip()
            latitude = _float(row.get("LATITUDE"))
            longitude = _float(row.get("LONGITUDE"))
            carbon = _float(row.get("CARBON_CONTENT_G_KG"))
            if (
                target_kind not in DIRECT_TARGET_KINDS
                or not method
                or normalized_method
                in {_normalize(NOT_REPORTED), _normalize(NOT_APPLICABLE_PSEUDO)}
                or latitude is None
                or longitude is None
                or carbon is None
            ):
                continue
            observations.append(
                MethodObservation(
                    source_id=(row.get("SOURCE_ID") or "").strip(),
                    dataset_id=(row.get("DATASET_ID") or "").strip(),
                    method=method,
                    latitude=latitude,
                    longitude=longitude,
                    sample_year=(row.get("SAMPLE_YEAR") or "").strip(),
                    depth_top_cm=_float(row.get("DEPTH_TOP_CM")),
                    depth_bottom_cm=_float(row.get("DEPTH_BOTTOM_CM")),
                    carbon_g_kg=carbon,
                )
            )
    return observations


def _depth_bucket(row: MethodObservation) -> str | None:
    if row.depth_top_cm is None and row.depth_bottom_cm is None:
        return None
    if row.depth_top_cm is None:
        midpoint = row.depth_bottom_cm
    elif row.depth_bottom_cm is None:
        midpoint = row.depth_top_cm
    else:
        midpoint = (row.depth_top_cm + row.depth_bottom_cm) / 2
    if midpoint is None or midpoint < 0:
        return None
    if midpoint <= 5:
        return "0-5"
    if midpoint <= 15:
        return "5-15"
    if midpoint <= 30:
        return "15-30"
    if midpoint <= 60:
        return "30-60"
    if midpoint <= 100:
        return "60-100"
    return ">100"


def aggregate_point_year(
    observations: Iterable[MethodObservation],
    include_depth: bool = False,
) -> dict[tuple[str, ...], float]:
    grouped: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in observations:
        key = (
            row.method,
            f"{round(row.latitude, 5):.5f}",
            f"{round(row.longitude, 5):.5f}",
            row.sample_year,
        )
        if include_depth:
            depth = _depth_bucket(row)
            if depth is None:
                continue
            key = (*key, depth)
        grouped[key].append(row.carbon_g_kg)
    return {key: float(np.median(values)) for key, values in grouped.items()}


def _epsilon_squared(h_statistic: float, groups: int, observations: int) -> float:
    denominator = observations - groups
    if denominator <= 0:
        return math.nan
    return max(0.0, (h_statistic - groups + 1) / denominator)


def _holm_adjust(p_values: Sequence[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [1.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (len(p_values) - rank) * float(p_values[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def _cliffs_delta(u_statistic: float, n_a: int, n_b: int) -> float:
    return 2 * float(u_statistic) / (n_a * n_b) - 1


def _effect_label(delta: float) -> str:
    magnitude = abs(delta)
    if magnitude < 0.147:
        return "desprezível"
    if magnitude < 0.33:
        return "pequeno"
    if magnitude < 0.474:
        return "moderado"
    return "grande"


def _method_family(method: str) -> str:
    normalized = _normalize(method)
    if normalized.startswith("combustao seca"):
        return "dry_combustion"
    if normalized.startswith("oxidacao umida"):
        return "wet_oxidation"
    return "other"


def _short_method(method: str) -> str:
    replacements = (
        ("Combustão seca - ", "Seca: "),
        ("Oxidação úmida - ", "Úmida: "),
    )
    for prefix, replacement in replacements:
        if method.startswith(prefix):
            return replacement + method[len(prefix) :]
    return method


def method_statistics(
    observations: Sequence[MethodObservation],
    point_year: dict[tuple[str, ...], float],
) -> tuple[list[dict[str, object]], dict[str, str]]:
    raw_by_method: dict[str, list[float]] = defaultdict(list)
    point_by_method: dict[str, list[float]] = defaultdict(list)
    sources: dict[str, set[str]] = defaultdict(set)
    datasets: dict[str, set[str]] = defaultdict(set)
    for row in observations:
        raw_by_method[row.method].append(row.carbon_g_kg)
        sources[row.method].add(row.source_id)
        datasets[row.method].add(row.dataset_id)
    for key, value in point_year.items():
        point_by_method[key[0]].append(value)

    ordered_methods = sorted(
        point_by_method,
        key=lambda method: (-float(np.median(point_by_method[method])), method),
    )
    codes = {method: f"M{index:02d}" for index, method in enumerate(ordered_methods, 1)}
    rows: list[dict[str, object]] = []
    for method in ordered_methods:
        values = np.asarray(point_by_method[method], dtype=float)
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        rows.append(
            {
                "method_code": codes[method],
                "carbon_analysis_method": method,
                "method_family": _method_family(method),
                "raw_rows": len(raw_by_method[method]),
                "point_year_observations": len(values),
                "source_count": len(sources[method]),
                "dataset_count": len(datasets[method]),
                "mean_g_kg": float(np.mean(values)),
                "q1_g_kg": float(q1),
                "median_g_kg": float(median),
                "q3_g_kg": float(q3),
                "minimum_g_kg": float(np.min(values)),
                "maximum_g_kg": float(np.max(values)),
            }
        )
    return rows, codes


def pairwise_statistics(
    point_year: dict[tuple[str, ...], float],
    codes: dict[str, str],
) -> list[dict[str, object]]:
    values: dict[str, np.ndarray] = {}
    for method in codes:
        values[method] = np.asarray(
            [value for key, value in point_year.items() if key[0] == method],
            dtype=float,
        )
    rows: list[dict[str, object]] = []
    raw_p_values: list[float] = []
    for method_a, method_b in combinations(codes, 2):
        a = values[method_a]
        b = values[method_b]
        result = stats.mannwhitneyu(a, b, alternative="two-sided", method="asymptotic")
        delta = _cliffs_delta(float(result.statistic), len(a), len(b))
        raw_p_values.append(float(result.pvalue))
        rows.append(
            {
                "method_a_code": codes[method_a],
                "method_a": method_a,
                "method_b_code": codes[method_b],
                "method_b": method_b,
                "n_a": len(a),
                "n_b": len(b),
                "median_a_g_kg": float(np.median(a)),
                "median_b_g_kg": float(np.median(b)),
                "median_difference_a_minus_b_g_kg": float(np.median(a) - np.median(b)),
                "mann_whitney_u": float(result.statistic),
                "mann_whitney_p": float(result.pvalue),
                "cliffs_delta_a_over_b": delta,
                "effect_magnitude": _effect_label(delta),
            }
        )
    for row, adjusted in zip(rows, _holm_adjust(raw_p_values)):
        row["mann_whitney_p_holm"] = adjusted
        row["significant_holm_0_05"] = adjusted < 0.05
    return rows


def depth_statistics(
    observations: Sequence[MethodObservation],
) -> list[dict[str, object]]:
    aggregated = aggregate_point_year(observations, include_depth=True)
    rows: list[dict[str, object]] = []
    for depth in DEPTH_ORDER:
        by_method: dict[str, list[float]] = defaultdict(list)
        for key, value in aggregated.items():
            if key[-1] == depth:
                by_method[key[0]].append(value)
        arrays = [
            np.asarray(values, dtype=float)
            for values in by_method.values()
            if len(values) >= 5
        ]
        if len(arrays) < 2:
            continue
        result = stats.kruskal(*arrays)
        total = sum(len(array) for array in arrays)
        rows.append(
            {
                "depth_band_cm": depth,
                "point_year_depth_observations": total,
                "methods_with_at_least_5_observations": len(arrays),
                "kruskal_wallis_h": float(result.statistic),
                "degrees_of_freedom": len(arrays) - 1,
                "p_value": float(result.pvalue),
                "epsilon_squared": _epsilon_squared(
                    float(result.statistic), len(arrays), total
                ),
            }
        )
    return rows


def _global_test(values_by_method: Sequence[np.ndarray]) -> dict[str, object]:
    result = stats.kruskal(*values_by_method)
    count = sum(len(values) for values in values_by_method)
    groups = len(values_by_method)
    return {
        "observations": count,
        "methods": groups,
        "kruskal_wallis_h": float(result.statistic),
        "degrees_of_freedom": groups - 1,
        "p_value": float(result.pvalue),
        "epsilon_squared": _epsilon_squared(float(result.statistic), groups, count),
    }


def _p_text(value: float) -> str:
    if value < 0.001:
        exponent = int(math.floor(math.log10(value))) if value > 0 else -999
        if exponent <= -100:
            return "p < 10^-100"
        return f"p = {value:.1e}"
    return f"p = {value:.3f}".replace(".", ",")


def _plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 14,
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


def plot_significance_figure(
    method_rows: Sequence[dict[str, object]],
    pairwise: Sequence[dict[str, object]],
    depth_rows: Sequence[dict[str, object]],
    global_raw: dict[str, object],
    global_point_year: dict[str, object],
    output_path: Path,
) -> None:
    _plot_style()
    codes = [str(row["method_code"]) for row in method_rows]
    code_index = {code: index for index, code in enumerate(codes)}
    size = len(codes)
    fig = plt.figure(figsize=(22, 19))
    grid = fig.add_gridspec(
        3,
        2,
        height_ratios=[1, 1, 0.29],
        hspace=0.34,
        wspace=0.25,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])
    key_ax = fig.add_subplot(grid[2, :])

    positions = np.arange(size)
    medians = np.asarray([float(row["median_g_kg"]) for row in method_rows])
    q1 = np.asarray([float(row["q1_g_kg"]) for row in method_rows])
    q3 = np.asarray([float(row["q3_g_kg"]) for row in method_rows])
    colors = [FAMILY_COLORS[str(row["method_family"])] for row in method_rows]
    for position, median, low, high, color in zip(positions, medians, q1, q3, colors):
        ax_a.plot([low, high], [position, position], color=color, linewidth=6, solid_capstyle="round")
        ax_a.scatter(
            median,
            position,
            s=66,
            facecolor="white",
            edgecolor=color,
            linewidth=2,
            zorder=3,
        )
    ax_a.set_yticks(
        positions,
        [
            f"{row['method_code']}  n={int(row['point_year_observations'])}"
            for row in method_rows
        ],
    )
    ax_a.invert_yaxis()
    if np.all(q1 > 0):
        ax_a.set_xscale("log")
        ax_a.set_xlabel("Concentração de carbono (g/kg), escala log")
    else:
        ax_a.set_xscale("symlog", linthresh=1)
        ax_a.set_xlabel("Concentração de carbono (g/kg), escala symlog")
    ax_a.set_title("a) Centro e dispersão por método", loc="left", fontweight="bold")
    ax_a.grid(axis="y", visible=False)
    ax_a.spines[["top", "right", "left"]].set_visible(False)
    ax_a.tick_params(axis="y", length=0)
    ax_a.legend(
        handles=[
            Patch(facecolor=FAMILY_COLORS[family], label=FAMILY_LABELS[family])
            for family in ("dry_combustion", "wet_oxidation")
        ],
        loc="lower right",
        fontsize=9,
    )
    ax_a.text(
        0,
        -0.12,
        "Ponto = mediana; barra = Q1-Q3; n = pontos-ano.",
        transform=ax_a.transAxes,
        color="#59666F",
        fontsize=9,
    )

    delta_matrix = np.zeros((size, size), dtype=float)
    for row in pairwise:
        i = code_index[str(row["method_a_code"])]
        j = code_index[str(row["method_b_code"])]
        delta = float(row["cliffs_delta_a_over_b"])
        delta_matrix[i, j] = delta
        delta_matrix[j, i] = -delta
    delta_cmap = LinearSegmentedColormap.from_list(
        "blue_orange",
        ["#2F6B9A", "#F7F8F8", "#C58B2A"],
    )
    delta_image = ax_b.imshow(
        delta_matrix,
        cmap=delta_cmap,
        norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1),
        aspect="equal",
    )
    ax_b.set_xticks(range(size), codes, rotation=45, ha="right")
    ax_b.set_yticks(range(size), codes)
    ax_b.tick_params(length=0, labelsize=8)
    for i in range(size):
        for j in range(size):
            if i == j:
                label = "-"
            else:
                label = f"{delta_matrix[i, j]:+.2f}"
            ax_b.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=5.7,
                color="white" if abs(delta_matrix[i, j]) > 0.58 else "#172126",
            )
    ax_b.grid(False)
    ax_b.set_title("b) Tamanho de efeito entre pares", loc="left", fontweight="bold")
    colorbar_b = fig.colorbar(delta_image, ax=ax_b, fraction=0.045, pad=0.03)
    colorbar_b.set_label("Delta de Cliff: método da linha versus coluna")
    ax_b.text(
        0,
        -0.18,
        "Sinal positivo indica concentrações maiores no método da linha; |delta| >= 0,474 é grande.",
        transform=ax_b.transAxes,
        color="#59666F",
        fontsize=9,
    )

    p_matrix = np.full((size, size), np.nan, dtype=float)
    significant = np.zeros((size, size), dtype=bool)
    for row in pairwise:
        i = code_index[str(row["method_a_code"])]
        j = code_index[str(row["method_b_code"])]
        p_value = float(row["mann_whitney_p_holm"])
        p_matrix[i, j] = p_value
        p_matrix[j, i] = p_value
        significant[i, j] = significant[j, i] = p_value < 0.05
    strength = np.minimum(-np.log10(np.clip(p_matrix, 1e-300, 1)), 20)
    masked_strength = np.ma.masked_where(~significant, strength)
    significance_cmap = matplotlib.colormaps["Blues"].copy()
    significance_cmap.set_bad("#E8ECEE")
    p_image = ax_c.imshow(masked_strength, cmap=significance_cmap, vmin=1.301, vmax=20)
    ax_c.set_xticks(range(size), codes, rotation=45, ha="right")
    ax_c.set_yticks(range(size), codes)
    ax_c.tick_params(length=0, labelsize=8)
    for i in range(size):
        for j in range(size):
            if i == j:
                label = "-"
            elif significant[i, j]:
                p_value = p_matrix[i, j]
                label = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*"
            else:
                label = "ns"
            dark = significant[i, j] and strength[i, j] > 8
            ax_c.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=6.4,
                color="white" if dark else "#172126",
                fontweight="bold" if significant[i, j] else "normal",
            )
    ax_c.grid(False)
    significant_pairs = sum(bool(row["significant_holm_0_05"]) for row in pairwise)
    ax_c.set_title(
        f"c) Significância pareada: {significant_pairs}/{len(pairwise)} pares",
        loc="left",
        fontweight="bold",
    )
    colorbar_c = fig.colorbar(p_image, ax=ax_c, fraction=0.045, pad=0.03)
    colorbar_c.set_label("-log10(p ajustado por Holm), máximo 20")
    ax_c.text(
        0,
        -0.18,
        "Mann-Whitney bilateral: * p<0,05; ** p<0,01; *** p<0,001; ns = não significativo.",
        transform=ax_c.transAxes,
        color="#59666F",
        fontsize=9,
    )

    depth_lookup = {str(row["depth_band_cm"]): row for row in depth_rows}
    plotted_depths = [depth for depth in DEPTH_ORDER if depth in depth_lookup]
    effects = [float(depth_lookup[depth]["epsilon_squared"]) for depth in plotted_depths]
    y = np.arange(len(plotted_depths))
    bars = ax_d.barh(
        y,
        effects,
        color="#C58B2A",
        edgecolor="#815A1B",
        linewidth=0.8,
        height=0.62,
    )
    ax_d.set_yticks(y, [f"{depth} cm" for depth in plotted_depths])
    ax_d.invert_yaxis()
    ax_d.axvline(0.14, color="#4F5B62", linestyle="--", linewidth=1.2)
    ax_d.text(
        0.14,
        1.01,
        "efeito grande (0,14)",
        transform=ax_d.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=8,
        color="#4F5B62",
    )
    for bar, depth in zip(bars, plotted_depths):
        row = depth_lookup[depth]
        effect = float(row["epsilon_squared"])
        ax_d.text(
            effect + 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{effect:.3f} | {_p_text(float(row['p_value']))}",
            va="center",
            fontsize=8.5,
            color="#263238",
        )
    ax_d.set_xlim(0, max([0.78, *(effect + 0.22 for effect in effects)]))
    ax_d.set_xlabel("Epsilon quadrado de Kruskal-Wallis")
    ax_d.set_title("d) Robustez por faixa de profundidade", loc="left", fontweight="bold")
    ax_d.grid(axis="y", visible=False)
    ax_d.spines[["top", "right", "left"]].set_visible(False)
    ax_d.tick_params(axis="y", length=0)
    ax_d.text(
        0,
        1.12,
        (
            f"Todas as camadas: ε²={float(global_raw['epsilon_squared']):.3f}, "
            f"{_p_text(float(global_raw['p_value']))}  |  "
            f"Pontos-ano: ε²={float(global_point_year['epsilon_squared']):.3f}, "
            f"{_p_text(float(global_point_year['p_value']))}"
        ),
        transform=ax_d.transAxes,
        fontsize=9.3,
        color="#263238",
        fontweight="bold",
    )
    ax_d.text(
        0,
        -0.12,
        "Dentro de cada faixa, camadas do mesmo ponto-ano e método foram agregadas pela mediana.",
        transform=ax_d.transAxes,
        color="#59666F",
        fontsize=9,
    )

    key_ax.axis("off")
    key_ax.text(
        0,
        1.03,
        "Chave dos métodos (n = pontos-ano)",
        transform=key_ax.transAxes,
        fontsize=11,
        fontweight="bold",
        color="#172126",
    )
    columns = 3
    rows_per_column = math.ceil(size / columns)
    for index, row in enumerate(method_rows):
        column = index // rows_per_column
        row_in_column = index % rows_per_column
        x = column / columns
        y_position = 0.82 - row_in_column * 0.23
        family = str(row["method_family"])
        label = _short_method(str(row["carbon_analysis_method"]))
        label = "\n".join(textwrap.wrap(label, width=48))
        key_ax.text(
            x,
            y_position,
            str(row["method_code"]),
            transform=key_ax.transAxes,
            va="top",
            fontsize=9,
            fontweight="bold",
            color=FAMILY_COLORS[family],
        )
        key_ax.text(
            x + 0.032,
            y_position,
            f"{label}  [n={int(row['point_year_observations'])}]",
            transform=key_ax.transAxes,
            va="top",
            fontsize=8.4,
            color="#263238",
            linespacing=1.1,
        )

    fig.suptitle(
        "Significância das diferenças de concentração de carbono entre métodos",
        x=0.04,
        y=0.985,
        ha="left",
        fontsize=20,
        fontweight="bold",
        color="#172126",
    )
    fig.text(
        0.04,
        0.959,
        (
            "Somente concentrações diretamente medidas. Inferência principal baseada na "
            f"mediana de {int(global_point_year['observations'])} pontos-ano em {size} métodos."
        ),
        ha="left",
        fontsize=11,
        color="#52616A",
    )
    fig.text(
        0.04,
        0.012,
        (
            "Interpretação: as distribuições rotuladas por método diferem, mas o efeito não é causal. "
            "Método, fonte, estudo, região e tipo de solo estão confundidos; grupos muito pequenos exigem cautela."
        ),
        ha="left",
        fontsize=9.2,
        color="#59666F",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def build_method_significance_figure(
    input_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    observations = load_direct_method_observations(input_path)
    if not observations:
        raise ValueError("No directly measured carbon concentrations with a reported method")
    point_year = aggregate_point_year(observations)
    method_rows, codes = method_statistics(observations, point_year)
    if len(method_rows) < 2:
        raise ValueError("At least two analytical methods are required")
    pairwise = pairwise_statistics(point_year, codes)
    depth_rows = depth_statistics(observations)

    raw_by_method = [
        np.asarray(
            [row.carbon_g_kg for row in observations if row.method == method],
            dtype=float,
        )
        for method in codes
    ]
    point_by_method = [
        np.asarray(
            [value for key, value in point_year.items() if key[0] == method],
            dtype=float,
        )
        for method in codes
    ]
    global_raw = _global_test(raw_by_method)
    global_point_year = _global_test(point_by_method)

    output_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = output_dir / "graficos"
    charts_dir.mkdir(parents=True, exist_ok=True)
    method_csv = output_dir / "significancia_metodos_resumo.csv"
    pairwise_csv = output_dir / "significancia_metodos_pares.csv"
    depth_csv = output_dir / "significancia_metodos_profundidade.csv"
    summary_json = output_dir / "significancia_metodos_resumo.json"
    figure_path = charts_dir / "11_significancia_metodos_carbono.png"

    _write_csv(method_csv, method_rows)
    _write_csv(pairwise_csv, pairwise)
    _write_csv(depth_csv, depth_rows)
    summary = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "input_file": str(input_path.resolve()),
        "direct_measurement_rows": len(observations),
        "point_year_observations": len(point_year),
        "methods": len(method_rows),
        "pairwise_comparisons": len(pairwise),
        "significant_pairwise_holm_0_05": sum(
            bool(row["significant_holm_0_05"]) for row in pairwise
        ),
        "global_all_layers": global_raw,
        "global_point_year": global_point_year,
        "method_summary_csv": str(method_csv.resolve()),
        "pairwise_csv": str(pairwise_csv.resolve()),
        "depth_csv": str(depth_csv.resolve()),
        "figure": str(figure_path.resolve()),
    }
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    plot_significance_figure(
        method_rows,
        pairwise,
        depth_rows,
        global_raw,
        global_point_year,
        figure_path,
    )
    return summary
