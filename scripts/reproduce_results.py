from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from carbono_solo.analysis.carbon_method_significance import (
    build_method_significance_figure,
)
from carbono_solo.analysis.carbon_methods import build_carbon_method_figures
from carbono_solo.analysis.carbon_sources import run_source_comparison
from carbono_solo.analysis.manuscript_assets import build_manuscript_assets
from carbono_solo.collector.depth_groups import group_stock_by_depth_csv


BASELINE_FILES = (
    "soil_targets_brasil.csv",
    "soil_targets_duplicates_brasil.csv",
    "soil_targets_overlap_audit_brasil.csv",
    "soil_targets_brasil_harmonized.csv",
    "soil_targets_brasil_harmonization_audit.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the statistical outputs, tables, and figures used in the manuscript.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing the versioned baseline CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/reproduction"),
        help="New directory for regenerated outputs.",
    )
    parser.add_argument(
        "--state-geojson",
        type=Path,
        default=Path("data/ibge-brasil-ufs.geojson"),
    )
    parser.add_argument(
        "--biome-shapefile",
        type=Path,
        default=Path("data/raw/ibge/biomes/2025/lml_bioma_e250k_v20250911_A.shp"),
    )
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def main() -> int:
    args = parse_args()
    input_dir = resolve_project_path(args.input_dir).resolve()
    output_dir = resolve_project_path(args.output_dir).resolve()
    state_geojson = resolve_project_path(args.state_geojson).resolve()
    biome_shapefile = resolve_project_path(args.biome_shapefile).resolve()

    required = [input_dir / name for name in BASELINE_FILES]
    required.extend((state_geojson, biome_shapefile))
    missing = [path for path in required if not path.exists()]
    if missing:
        raise SystemExit("Missing required inputs:\n" + "\n".join(str(path) for path in missing))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"Output directory is not empty: {output_dir}")

    processed_dir = output_dir / "processed"
    comparison_dir = output_dir / "comparison"
    manuscript_dir = output_dir / "manuscript"
    processed_dir.mkdir(parents=True, exist_ok=True)

    input_hashes: dict[str, str] = {}
    for source in required[: len(BASELINE_FILES)]:
        destination = processed_dir / source.name
        shutil.copy2(source, destination)
        input_hashes[source.name] = sha256(source)

    harmonized = processed_dir / "soil_targets_brasil_harmonized.csv"
    depth_summary = group_stock_by_depth_csv(
        harmonized,
        processed_dir / "soil_target_Brasil_Stock_Group_Depth.csv",
        processed_dir / "soil_target_Brasil_Stock_Group_Depth_audit.csv",
    )
    source_summary = run_source_comparison(harmonized, state_geojson, comparison_dir)
    method_summary = build_carbon_method_figures(harmonized, comparison_dir)
    significance_summary = build_method_significance_figure(harmonized, comparison_dir)
    manuscript_summary = build_manuscript_assets(
        harmonized,
        processed_dir,
        comparison_dir,
        manuscript_dir,
        state_geojson,
        biome_shapefile,
    )

    summary = {
        "inputs_sha256": input_hashes,
        "depth_harmonization": depth_summary,
        "source_comparison": source_summary,
        "analytical_methods": method_summary,
        "method_significance": significance_summary,
        "manuscript_assets": manuscript_summary,
    }
    summary_path = output_dir / "reproduction_summary.json"
    summary_path.write_text(
        json.dumps(json_ready(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Reproduction complete: {output_dir}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
