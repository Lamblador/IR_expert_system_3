from pathlib import Path

from ir_pipeline.bands import BandDef, load_bands, structure_matches_band
from rdkit import Chem


def test_load_bands_yaml():
    bands = load_bands(Path("configs/bands_reference.yaml"))
    assert len(bands) >= 10
    assert all(isinstance(b, BandDef) for b in bands)


def test_smarts_carbonyl_matches_acetone():
    bands = load_bands(Path("configs/bands_reference.yaml"))
    mol = Chem.MolFromSmiles("CC(=O)C")
    cb = next(b for b in bands if b.band_id == "carbonyl")
    assert structure_matches_band(mol, cb)


def test_smarts_nitrile_matches():
    bands = load_bands(Path("configs/bands_reference.yaml"))
    mol = Chem.MolFromSmiles("CC#N")
    nb = next(b for b in bands if b.band_id == "nitrile")
    assert structure_matches_band(mol, nb)
