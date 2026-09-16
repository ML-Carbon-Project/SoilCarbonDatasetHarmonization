from __future__ import annotations

import argparse
from pathlib import Path

from carbono_solo.collector.depth_groups import group_stock_by_depth_csv


DEFAULT_INPUT = Path("data/processed/soil_targets_brasil_harmonized.csv")
DEFAULT_OUTPUT = Path("data/processed/soil_target_Brasil_Stock_Group_Depth.csv")
DEFAULT_AUDIT_OUTPUT = Path(
    "data/processed/soil_target_Brasil_Stock_Group_Depth_audit.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Group harmonized soil carbon stock targets by depth bucket.",
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        type=Path,
        help=f"Harmonized input CSV. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        type=Path,
        help=f"Grouped output CSV. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--audit-output",
        default=DEFAULT_AUDIT_OUTPUT,
        type=Path,
        help=f"Depth harmonization audit CSV. Default: {DEFAULT_AUDIT_OUTPUT}",
    )
    parser.add_argument(
        "--spline-lambda",
        default=0.1,
        type=float,
        help="Equal-area spline smoothing parameter. Default: 0.1",
    )
    parser.add_argument(
        "--max-spline-mass-error-pct",
        default=10.0,
        type=float,
        help="Maximum pre-rescale spline mass error. Default: 10 percent",
    )
    parser.add_argument(
        "--no-splines",
        action="store_true",
        help="Use mass-preserving overlap allocation for every profile.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = group_stock_by_depth_csv(
        args.input,
        args.output,
        args.audit_output,
        spline_lambda=args.spline_lambda,
        max_spline_mass_error_pct=args.max_spline_mass_error_pct,
        use_splines=not args.no_splines,
    )
    print("Mass-preserving depth harmonization complete")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"output: {args.output.resolve()}")
    print(f"audit_output: {args.audit_output.resolve()}")


if __name__ == "__main__":
    main()
