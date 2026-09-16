from __future__ import annotations

import argparse
from pathlib import Path

from carbono_solo.collector.biomes import (
    enrich_biome_csv,
    ensure_ibge_biome_layer,
    ensure_ibge_territory_layer,
)
from carbono_solo.collector.harmonize import harmonize_csv


DEFAULT_INPUT = Path("data/processed/soil_targets_brasil.csv")
DEFAULT_OUTPUT = Path("data/processed/soil_targets_brasil_harmonized.csv")
DEFAULT_HARMONIZATION_AUDIT = Path(
    "data/processed/soil_targets_brasil_harmonization_audit.csv"
)
DEFAULT_BIOME_AUDIT = Path("data/processed/soil_targets_brasil_biome_audit.csv")
DEFAULT_BIOME_LAYER_DIR = Path("data/raw/ibge/biomes/2025")
DEFAULT_TERRITORY_LAYER_DIR = Path("data/raw/ibge/territory/2025")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Harmonize soil carbon target rows and standardize stock to Mg/ha.",
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        type=Path,
        help=f"Input target CSV. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        type=Path,
        help=f"Harmonized output CSV. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--harmonization-audit",
        default=DEFAULT_HARMONIZATION_AUDIT,
        type=Path,
        help=(
            "Row-level harmonization exclusion audit CSV. "
            f"Default: {DEFAULT_HARMONIZATION_AUDIT}"
        ),
    )
    parser.add_argument(
        "--biome-audit",
        default=DEFAULT_BIOME_AUDIT,
        type=Path,
        help=f"Biome/territory audit CSV. Default: {DEFAULT_BIOME_AUDIT}",
    )
    parser.add_argument(
        "--biome-layer-dir",
        default=DEFAULT_BIOME_LAYER_DIR,
        type=Path,
        help=f"Pinned IBGE biome layer directory. Default: {DEFAULT_BIOME_LAYER_DIR}",
    )
    parser.add_argument(
        "--territory-layer-dir",
        default=DEFAULT_TERRITORY_LAYER_DIR,
        type=Path,
        help=(
            "Pinned IBGE Brazil territory layer directory. "
            f"Default: {DEFAULT_TERRITORY_LAYER_DIR}"
        ),
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Require the pinned IBGE archives to be available locally.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    harmonization_summary = harmonize_csv(
        args.input,
        args.output,
        args.harmonization_audit,
    )
    print("Harmonization complete")
    for key, value in harmonization_summary.items():
        print(f"{key}: {value}")

    biome_layer = ensure_ibge_biome_layer(
        args.biome_layer_dir,
        allow_download=not args.offline,
    )
    territory_layer = ensure_ibge_territory_layer(
        args.territory_layer_dir,
        allow_download=not args.offline,
    )
    enrichment_summary = enrich_biome_csv(
        args.output,
        args.output,
        args.biome_audit,
        biome_layer,
        territory_layer,
    )
    print("Territory and biome enrichment complete")
    for key, value in enrichment_summary.items():
        print(f"{key}: {value}")
    print(f"output: {args.output.resolve()}")
    print(f"harmonization_audit: {args.harmonization_audit.resolve()}")
    print(f"biome_audit: {args.biome_audit.resolve()}")


if __name__ == "__main__":
    main()
