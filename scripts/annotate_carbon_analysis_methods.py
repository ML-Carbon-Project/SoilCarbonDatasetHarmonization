from __future__ import annotations

import argparse
from pathlib import Path

from carbono_solo.collector.lab_methods import annotate_target_csvs


DEFAULT_ACTIVE = Path("data/processed/soil_targets_brasil.csv")
DEFAULT_DUPLICATES = Path("data/processed/soil_targets_duplicates_brasil.csv")
DEFAULT_AUDIT = Path("data/processed/carbon_analysis_method_inventory.csv")
DEFAULT_HYBRAS = Path("data/raw/hybras/v1_2020/HYBRAS_COMPLETE_V1.xlsx")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate soil target rows with documented carbon analysis methods.",
    )
    parser.add_argument("--active", type=Path, default=DEFAULT_ACTIVE)
    parser.add_argument("--duplicates", type=Path, default=DEFAULT_DUPLICATES)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--hybras-workbook", type=Path, default=DEFAULT_HYBRAS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = annotate_target_csvs(
        args.active,
        args.duplicates,
        args.audit,
        args.hybras_workbook,
    )
    print("Carbon analysis method annotation complete")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"audit: {args.audit.resolve()}")


if __name__ == "__main__":
    main()
