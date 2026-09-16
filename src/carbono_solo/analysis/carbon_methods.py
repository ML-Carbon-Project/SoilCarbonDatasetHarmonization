from __future__ import annotations

import csv
import os
import tempfile
import textwrap
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "carbono_solo_matplotlib"),
)

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch

from carbono_solo.collector.io import write_csv
from carbono_solo.collector.lab_methods import NOT_APPLICABLE_PSEUDO, NOT_REPORTED


SOURCE_LABELS = {
    "mapbiomas_soc": "MapBiomas Solo C3",
    "wosis_brazil": "WoSIS",
    "bdsolos_embrapa": "BD Solos / Embrapa",
    "soildata_dataset": "SoilData",
    "israd_brazil": "ISRaD",
    "hybras_sgb": "HYBRAS / SGB",
}

FAMILY_LABELS = {
    "not_reported": "N\u00e3o informado",
    "not_applicable": "N\u00e3o se aplica",
    "dry_combustion": "Combust\u00e3o seca",
    "wet_oxidation": "Oxida\u00e7\u00e3o \u00famida",
    "specific": "M\u00e9todo espec\u00edfico",
}

FAMILY_COLORS = {
    "not_reported": "#6C757D",
    "not_applicable": "#C58B2A",
    "dry_combustion": "#2F6B9A",
    "wet_oxidation": "#2F855A",
    "specific": "#9C5B45",
}

GENERAL_FIELDS = [
    "carbon_analysis_method",
    "row_count",
    "percentage",
]

SOURCE_DATASET_FIELDS = [
    "source_id",
    "dataset_id",
    "carbon_analysis_method",
    "row_count",
    "percentage_within_source_dataset",
]


@dataclass(frozen=True)
class CarbonMethodRow:
    source_id: str
    dataset_id: str
    method: str


def load_carbon_method_rows(path: Path) -> list[CarbonMethodRow]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        required = {"SOURCE_ID", "DATASET_ID", "CARBON_ANALYSIS_METHOD"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Missing required harmonized columns: {sorted(missing)}"
            )
        return [
            CarbonMethodRow(
                source_id=(row.get("SOURCE_ID") or "").strip(),
                dataset_id=(row.get("DATASET_ID") or "").strip(),
                method=(row.get("CARBON_ANALYSIS_METHOD") or "").strip()
                or NOT_REPORTED,
            )
            for row in reader
        ]


def summarize_carbon_methods(
    rows: Iterable[CarbonMethodRow],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    materialized = list(rows)
    total_rows = len(materialized)
    method_counts = Counter(row.method for row in materialized)
    source_totals = Counter((row.source_id, row.dataset_id) for row in materialized)
    source_method_counts = Counter(
        (row.source_id, row.dataset_id, row.method) for row in materialized
    )

    general = [
        {
            "carbon_analysis_method": method,
            "row_count": count,
            "percentage": 0.0 if total_rows == 0 else count / total_rows * 100,
        }
        for method, count in sorted(
            method_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]
    by_source_dataset = [
        {
            "source_id": source_id,
            "dataset_id": dataset_id,
            "carbon_analysis_method": method,
            "row_count": count,
            "percentage_within_source_dataset": (
                count / source_totals[(source_id, dataset_id)] * 100
            ),
        }
        for (source_id, dataset_id, method), count in sorted(
            source_method_counts.items(),
            key=lambda item: (
                -source_totals[(item[0][0], item[0][1])],
                item[0][0],
                item[0][1],
                -item[1],
                item[0][2],
            ),
        )
    ]
    return general, by_source_dataset


def plot_general_method_counts(
    general: list[dict[str, object]],
    output_path: Path,
) -> None:
    methods = [str(row["carbon_analysis_method"]) for row in general]
    counts = np.asarray([int(row["row_count"]) for row in general], dtype=float)
    percentages = [float(row["percentage"]) for row in general]
    families = [_method_family(method) for method in methods]
    colors = [FAMILY_COLORS[family] for family in families]

    figure_height = max(9.2, 0.62 * len(methods) + 3.2)
    fig, ax = plt.subplots(figsize=(18, figure_height))
    positions = np.arange(len(methods))
    bars = ax.barh(positions, counts, color=colors, height=0.7)
    ax.set_xscale("log")
    ax.set_xlim(0.8, max(counts) * 3.0)
    ax.set_yticks(
        positions,
        ["\n".join(textwrap.wrap(method, width=68)) for method in methods],
    )
    ax.invert_yaxis()
    ax.set_xlabel("Quantidade de linhas, escala logar\u00edtmica")
    ax.set_ylabel("")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color="#D7DDE0", linewidth=0.7, alpha=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=10)

    for bar, count, percentage in zip(bars, counts, percentages):
        label = f"{_pt_integer(int(count))}  ({_pt_decimal(percentage, 2)}%)"
        ax.text(
            count * 1.08,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            ha="left",
            fontsize=9.5,
            color="#263238",
            fontweight="bold" if percentage >= 1 else "normal",
        )

    used_families = [family for family in FAMILY_LABELS if family in set(families)]
    ax.legend(
        handles=[
            Patch(facecolor=FAMILY_COLORS[family], label=FAMILY_LABELS[family])
            for family in used_families
        ],
        loc="lower right",
        frameon=False,
        ncol=min(3, len(used_families)),
        fontsize=9,
    )
    fig.suptitle(
        "M\u00e9todos de an\u00e1lise de carbono no conjunto harmonizado",
        x=0.04,
        y=0.985,
        ha="left",
        fontsize=19,
        fontweight="bold",
        color="#172126",
    )
    fig.text(
        0.04,
        0.95,
        (
            f"Vis\u00e3o geral de {_pt_integer(int(np.sum(counts)))} linhas. "
            "As barras preservam as designa\u00e7\u00f5es laboratoriais espec\u00edficas."
        ),
        ha="left",
        fontsize=11,
        color="#52616A",
    )
    fig.text(
        0.04,
        0.012,
        (
            "Nota: o eixo logar\u00edtmico permite comparar categorias raras com "
            "as classes majorit\u00e1rias; os r\u00f3tulos mostram contagem e percentual exatos."
        ),
        ha="left",
        fontsize=9,
        color="#59666F",
    )
    fig.tight_layout(rect=(0.03, 0.05, 0.99, 0.93))
    _save_figure(fig, output_path)


def plot_source_dataset_method_counts(
    general: list[dict[str, object]],
    by_source_dataset: list[dict[str, object]],
    output_path: Path,
) -> None:
    method_order = [str(row["carbon_analysis_method"]) for row in general]
    source_totals = Counter()
    for row in by_source_dataset:
        source_totals[(str(row["source_id"]), str(row["dataset_id"]))] += int(
            row["row_count"]
        )
    source_dataset_order = sorted(
        source_totals,
        key=lambda key: (-source_totals[key], key[0], key[1]),
    )
    matrix = np.zeros((len(source_dataset_order), len(method_order)), dtype=float)
    row_index = {key: index for index, key in enumerate(source_dataset_order)}
    method_index = {method: index for index, method in enumerate(method_order)}
    for row in by_source_dataset:
        key = (str(row["source_id"]), str(row["dataset_id"]))
        method = str(row["carbon_analysis_method"])
        matrix[row_index[key], method_index[method]] = int(row["row_count"])

    masked = np.ma.masked_where(matrix == 0, matrix)
    color_map = matplotlib.colormaps["cividis"].copy()
    color_map.set_bad("#F1F3F2")
    maximum = max(float(np.max(matrix)), 1.0)
    norm = LogNorm(vmin=1, vmax=maximum)

    fig = plt.figure(figsize=(23, 12.5))
    grid = fig.add_gridspec(1, 2, width_ratios=[2.35, 1.15], wspace=0.08)
    ax = fig.add_subplot(grid[0, 0])
    key_ax = fig.add_subplot(grid[0, 1])
    image = ax.imshow(masked, cmap=color_map, norm=norm, aspect="auto")

    method_codes = [f"M{index:02d}" for index in range(1, len(method_order) + 1)]
    row_labels = [
        (
            f"{SOURCE_LABELS.get(source_id, source_id)} | {_dataset_label(dataset_id)}"
            f"\nn={_pt_integer(source_totals[(source_id, dataset_id)])}"
        )
        for source_id, dataset_id in source_dataset_order
    ]
    ax.set_xticks(range(len(method_order)), method_codes, rotation=0)
    ax.set_yticks(range(len(source_dataset_order)), row_labels)
    ax.tick_params(axis="both", length=0, labelsize=9)
    ax.set_xlabel("M\u00e9todo anal\u00edtico")
    ax.set_ylabel("Fonte | dataset")
    ax.grid(False)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = int(matrix[i, j])
            if value == 0:
                continue
            normalized_value = float(norm(float(value)))
            red, green, blue, _alpha = color_map(normalized_value)
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            ax.text(
                j,
                i,
                _pt_integer(value),
                ha="center",
                va="center",
                fontsize=7.5,
                color="#172126" if luminance > 0.58 else "white",
                fontweight="bold" if value >= 100 else "normal",
            )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.026, pad=0.018)
    colorbar.set_label("Quantidade de linhas, escala logar\u00edtmica")

    key_ax.axis("off")
    key_ax.text(
        0,
        1,
        "Chave dos m\u00e9todos",
        transform=key_ax.transAxes,
        va="top",
        fontsize=14,
        fontweight="bold",
        color="#172126",
    )
    y = 0.955
    line_step = min(0.061, 0.84 / max(len(method_order), 1))
    for code, row in zip(method_codes, general):
        method = str(row["carbon_analysis_method"])
        count = int(row["row_count"])
        wrapped = "\n".join(textwrap.wrap(method, width=56))
        key_ax.text(
            0,
            y,
            code,
            transform=key_ax.transAxes,
            va="top",
            fontsize=9.2,
            fontweight="bold",
            color=FAMILY_COLORS[_method_family(method)],
        )
        key_ax.text(
            0.075,
            y,
            f"{wrapped}  [n={_pt_integer(count)}]",
            transform=key_ax.transAxes,
            va="top",
            fontsize=8.4,
            color="#263238",
            linespacing=1.1,
        )
        y -= line_step

    fig.suptitle(
        "Quantidade de m\u00e9todos de an\u00e1lise por fonte e dataset",
        x=0.045,
        y=0.985,
        ha="left",
        fontsize=19,
        fontweight="bold",
        color="#172126",
    )
    fig.text(
        0.045,
        0.95,
        (
            "Cada c\u00e9lula mostra o n\u00famero de linhas. C\u00e9lulas vazias indicam "
            "aus\u00eancia daquele m\u00e9todo na combina\u00e7\u00e3o fonte/dataset."
        ),
        ha="left",
        fontsize=11,
        color="#52616A",
    )
    fig.text(
        0.045,
        0.012,
        (
            "A escala de cor \u00e9 logar\u00edtmica para manter vis\u00edveis datasets pequenos; "
            "os valores anotados s\u00e3o contagens absolutas."
        ),
        ha="left",
        fontsize=9,
        color="#59666F",
    )
    fig.subplots_adjust(left=0.18, right=0.98, top=0.91, bottom=0.08)
    _save_figure(fig, output_path)


def build_carbon_method_figures(
    input_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    rows = load_carbon_method_rows(input_path)
    general, by_source_dataset = summarize_carbon_methods(rows)
    charts_dir = output_dir / "graficos"
    charts_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    general_csv = output_dir / "metodos_carbono_resumo_geral.csv"
    source_dataset_csv = output_dir / "metodos_carbono_por_fonte_dataset.csv"
    general_figure = charts_dir / "09_metodos_carbono_visao_geral.png"
    source_dataset_figure = charts_dir / "10_metodos_carbono_por_fonte_dataset.png"

    write_csv(general_csv, GENERAL_FIELDS, general, encoding="utf-8-sig")
    write_csv(
        source_dataset_csv,
        SOURCE_DATASET_FIELDS,
        by_source_dataset,
        encoding="utf-8-sig",
    )
    plot_general_method_counts(general, general_figure)
    plot_source_dataset_method_counts(
        general,
        by_source_dataset,
        source_dataset_figure,
    )
    return {
        "input_rows": len(rows),
        "methods": len(general),
        "source_dataset_groups": len(
            {(row.source_id, row.dataset_id) for row in rows}
        ),
        "general_figure": general_figure,
        "source_dataset_figure": source_dataset_figure,
        "general_csv": general_csv,
        "source_dataset_csv": source_dataset_csv,
    }


def _method_family(method: str) -> str:
    if method == NOT_REPORTED:
        return "not_reported"
    if method == NOT_APPLICABLE_PSEUDO:
        return "not_applicable"
    normalized = _normalize(method)
    if normalized.startswith("combustao seca"):
        return "dry_combustion"
    if normalized.startswith("oxidacao umida"):
        return "wet_oxidation"
    return "specific"


def _dataset_label(dataset_id: str) -> str:
    if "/SoilData/" in dataset_id:
        return dataset_id.rsplit("/", 1)[-1]
    labels = {
        "wosis_latest_orgc": "latest ORGC",
        "bdsolos_public_carbono_organico": "carbono org\u00e2nico",
        "israd_flat_layer": "flat layer",
        "hybras_v1_2020": "v1.0",
    }
    return labels.get(dataset_id, dataset_id)


def _normalize(value: str) -> str:
    text = unicodedata.normalize("NFKD", value)
    return text.encode("ascii", "ignore").decode().lower()


def _pt_integer(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def _pt_decimal(value: float, digits: int) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
