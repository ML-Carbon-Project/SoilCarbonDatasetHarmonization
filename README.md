# Soil Carbon Dataset Harmonization

This repository contains the reproducible workflow and derived data used to harmonize legacy soil carbon observations across Brazil. The current release integrates six data sources into 65,769 provenance-traced soil-layer records and generates mass-preserving carbon-stock targets for standard depth intervals.

## Contents

- `data/processed`: harmonized CSV files and row-level audit tables.
- `data/raw/ibge`: IBGE boundaries required for territory and biome validation.
- `src/carbono_solo`: collection, deduplication, harmonization, and analysis code.
- `scripts`: executable workflow entry points.

## Duplicate reconciliation

Candidate overlaps are screened by coordinates, sampling period, depth, target type, and unit. A record is removed only when duplication is supported by an exact normalized scientific match, a shared source observation identifier without conflicting scientific fields, or documented HYBRAS/WoSIS republication. Discordant or unresolved overlaps remain active under a shared dependence-group identifier. The confirmed removals are stored in `soil_targets_duplicates_brasil.csv`, while `soil_targets_overlap_audit_brasil.csv` contains every candidate overlap, including ambiguous records retained for grouped validation and sensitivity analysis.

## Reproduction

Python 3.10 is required. Create the pinned environment and regenerate the statistical results, tables, and figures with:

```bash
python scripts/setup_env.py --no-dev
python scripts/reproduce_results.py
```

Generated outputs are written to `results/reproduction`. CSV files use a semicolon as the delimiter.

## Data terms

Redistributable derived data are versioned in this repository. Raw third-party files are not included when redistribution terms are source-specific or unspecified. The dataset inventory records source URLs, identifiers, and available license information; original source conditions remain applicable.
