from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from rdkit import Chem


@dataclass
class BandDef:
    band_id: str
    group: str
    atom_pair: str
    vibration_type: str
    range_min_cm1: float
    range_max_cm1: float
    typical_intensity: str
    intensity_reliability: str
    smarts: str | None
    applicability: str
    notes: str
    source: str
    pattern: Chem.Mol | None


def load_bands(yaml_path: Path) -> list[BandDef]:
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    bands: list[BandDef] = []
    for row in data.get("bands", []):
        sm = row.get("smarts")
        pat = Chem.MolFromSmarts(sm) if sm else None
        bands.append(
            BandDef(
                band_id=row["band_id"],
                group=row.get("group", ""),
                atom_pair=row.get("atom_pair", ""),
                vibration_type=row.get("vibration_type", ""),
                range_min_cm1=float(row["range_min_cm1"]),
                range_max_cm1=float(row["range_max_cm1"]),
                typical_intensity=str(row.get("typical_intensity", "")),
                intensity_reliability=str(row.get("intensity_reliability", "")),
                smarts=sm,
                applicability=str(row.get("applicability", "organic")),
                notes=str(row.get("notes", "")),
                source=str(row.get("source", "")),
                pattern=pat,
            )
        )
    return bands


def structure_matches_band(mol: Chem.Mol | None, band: BandDef) -> bool:
    if mol is None or band.pattern is None:
        return False
    try:
        return mol.HasSubstructMatch(band.pattern)
    except Exception:
        return False
