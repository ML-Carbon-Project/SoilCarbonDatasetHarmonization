from __future__ import annotations

import argparse
import json
from pathlib import Path

from carbono_solo.analysis.manuscript_assets import build_manuscript_assets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build figures and tables for the CATENA manuscript.")
    parser.add_argument("--harmonized", type=Path, default=Path("data/processed/soil_targets_brasil_harmonized.csv"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--comparison-dir", type=Path, default=Path("data/Compara\u00e7\u00e3o"))
    parser.add_argument("--manuscript-dir", type=Path, default=Path("data/Compara\u00e7\u00e3o/manuscrito"))
    parser.add_argument("--state-geojson", type=Path, default=Path("data/ibge-brasil-ufs.geojson"))
    parser.add_argument("--biome-shapefile", type=Path, default=Path("data/raw/ibge/biomes/2025/lml_bioma_e250k_v20250911_A.shp"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_manuscript_assets(
        args.harmonized,
        args.processed_dir,
        args.comparison_dir,
        args.manuscript_dir,
        args.state_geojson,
        args.biome_shapefile,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
