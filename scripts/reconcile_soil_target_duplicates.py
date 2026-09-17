from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from carbono_solo.collector.dedup import deduplicate_targets
from carbono_solo.collector.io import write_csv
from carbono_solo.collector.models import TARGET_FIELDS, TargetRecord


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconcile an existing Brazilian soil-target collection with the "
            "conservative duplicate policy."
        ),
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/processed"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed_dedup_stage"),
    )
    return parser.parse_args()


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=";"))


def _target_from_row(row: dict[str, str]) -> TargetRecord:
    values: dict[str, object] = {field: row.get(field, "") for field in TARGET_FIELDS}
    for field in ("latitude", "longitude", "coord_x", "coord_y"):
        values[field] = float(str(values[field]))
    return TargetRecord(**values)


def reconcile(input_dir: Path, output_dir: Path) -> dict[str, int]:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if input_dir == output_dir:
        raise ValueError("Input and output directories must be different.")

    active_path = input_dir / "soil_targets_brasil.csv"
    duplicates_path = input_dir / "soil_targets_duplicates_brasil.csv"
    rows = [*_read_rows(active_path), *_read_rows(duplicates_path)]
    records = [_target_from_row(row) for row in rows]
    resolved = deduplicate_targets(records)

    active = [row for row in resolved if row.duplicate_status == "active"]
    duplicates = [row for row in resolved if row.duplicate_status == "duplicate"]
    overlaps = [row for row in resolved if row.duplicate_group_id]
    ambiguous = [
        row
        for row in active
        if row.duplicate_resolution.startswith("retained_ambiguous_overlap:")
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = input_dir / "dataset_inventory_brasil.csv"
    if inventory_path.exists():
        shutil.copy2(inventory_path, output_dir / inventory_path.name)
    write_csv(
        output_dir / "soil_targets_brasil.csv",
        TARGET_FIELDS,
        (row.to_row() for row in active),
    )
    write_csv(
        output_dir / "soil_targets_duplicates_brasil.csv",
        TARGET_FIELDS,
        (row.to_row() for row in duplicates),
    )
    write_csv(
        output_dir / "soil_targets_overlap_audit_brasil.csv",
        TARGET_FIELDS,
        (row.to_row() for row in overlaps),
    )
    return {
        "input_targets": len(records),
        "active_targets": len(active),
        "confirmed_duplicate_targets": len(duplicates),
        "overlap_targets": len(overlaps),
        "ambiguous_targets": len(ambiguous),
    }


def main() -> int:
    args = parse_args()
    summary = reconcile(_absolute(args.input_dir), _absolute(args.output_dir))
    print("Conservative duplicate reconciliation complete")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"output_dir: {_absolute(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
