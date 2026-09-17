from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from carbono_solo.collector.runner import run_collection, utc_now_iso


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Brazilian soil carbon inventory and target CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory where CSV outputs are written.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Disable network-backed adapters.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    summary = run_collection(
        output_dir=output_dir,
        retrieved_at=utc_now_iso(),
        use_network=not args.offline,
    )

    print("Collection complete")
    for key in (
        "inventory_records",
        "active_targets",
        "duplicate_targets",
        "overlap_targets",
        "ambiguous_targets",
    ):
        print(f"{key}: {summary[key]}")
    print(f"output_dir: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
