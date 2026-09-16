from __future__ import annotations

import argparse
import json
from pathlib import Path

from carbono_solo.analysis.carbon_method_significance import (
    build_method_significance_figure,
)


DEFAULT_INPUT = Path("data/processed/soil_targets_brasil_harmonized.csv")
DEFAULT_OUTPUT = Path("data/Compara\u00e7\u00e3o")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot inferential comparisons among carbon analysis methods.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_method_significance_figure(args.input, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
