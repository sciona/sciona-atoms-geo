# sciona-atoms-geo

Geospatial provider repo for `sciona.atoms.*` namespace packages.

This repo contains atoms derived from geospatial and overhead-imagery
competition solutions. The initial families are extracted from the MIT-licensed
winning solutions for the 2021 Overhead Geopose Challenge.

## Current families

- **`geo/losses/`**: Geospatial regression and orientation loss functions.
- **`geo/augmentation/`**: GSD-aware image augmentations for overhead imagery.

## Design principles

1. **Extract the reusable geometry.** Atoms isolate the underlying geometric or
   statistical operation from the original training code and framework glue.

2. **Prefer framework-agnostic atoms.** Core atoms use numpy and OpenCV only.
   Framework-aware ports are added only when differentiability matters, such as
   training losses.

3. **Preserve provenance.** Every atom is paired with witnesses, CDG metadata,
   references, and review bundles tied back to the upstream competition source.

## Initial source

- **Source repo**:
  `kaggle-solutions/third_party_wave2/geopose-2021-winners/`
- **License**: MIT
- **Extracted from**:
  - `1st Place/training/losses.py`
  - `2nd Place/geopose/augmentations.py`

## Installation

```bash
pip install -e .
pip install -e '.[torch]'
```

## Directory structure

```text
src/sciona/atoms/geo/
├── augmentation/
│   ├── atoms.py
│   ├── cdg.json
│   ├── references.json
│   └── witnesses.py
└── losses/
    ├── atoms.py
    ├── atoms_torch.py
    ├── cdg.json
    ├── references.json
    └── witnesses.py
```

`oblique_angle_correction` was requested during ingestion, but no standalone
implementation of that operation was present in the referenced Geopose 2021
winner source file, so it is intentionally omitted from this initial ingest.
