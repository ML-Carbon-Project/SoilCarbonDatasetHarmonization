from __future__ import annotations

import argparse
import json
from pathlib import Path

from carbono_solo.analysis.carbon_sources import run_source_comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare carbon concentration across the six harmonized sources."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/soil_targets_brasil_harmonized.csv"),
    )
    parser.add_argument(
        "--geometry",
        type=Path,
        default=Path("data/ibge-brasil-ufs.geojson"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/Comparação"),
    )
    args = parser.parse_args()
    summary = run_source_comparison(args.input, args.geometry, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
