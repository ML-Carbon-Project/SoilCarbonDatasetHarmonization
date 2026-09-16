from __future__ import annotations

import argparse
from pathlib import Path

from carbono_solo.analysis.carbon_methods import build_carbon_method_figures


DEFAULT_INPUT = Path("data/processed/soil_targets_brasil_harmonized.csv")
DEFAULT_OUTPUT = Path("data/Compara\u00e7\u00e3o")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot carbon analysis method counts overall and by source/dataset.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_carbon_method_figures(args.input, args.output_dir)
    print("Carbon analysis method figures complete")
    for key, value in summary.items():
        if isinstance(value, Path):
            value = value.resolve()
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
