from pathlib import Path

import numpy as np
import pandas as pd

from ir_pipeline.dataset_merge import merge_dataset_versions


def _fake_dataset(root: Path, name: str, n: int, seed: int, prefix: str) -> None:
    rng = np.random.default_rng(seed)
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    wn = np.arange(400.0, 410.0, 2.0, dtype=np.float32)
    ids = [f"{prefix}{i:04d}" for i in range(n)]
    X = rng.random((n, wn.size), dtype=np.float32)
    np.savez_compressed(
        d / "spectra.npz",
        spectrum_id=np.asarray(ids, dtype=object),
        X=X,
        X_absorbance_corrected=X,
        X_absorbance_like_interp=X,
        coverage=np.ones((n, wn.size), dtype=np.uint8),
        wavenumbers=wn,
    )
    meta = pd.DataFrame(
        {
            "spectrum_id": ids,
            "title": [f"mol-{prefix}-{i}" for i in range(n)],
            "cas": [None] * n,
            "smiles": [f"CCO{i}" if i % 2 == 0 else None for i in range(n)],
            "inchikey": [f"KEY{prefix}{i}" if i % 2 == 0 else None for i in range(n)],
            "inchi": [None] * n,
            "qc_ok": True,
            "origin": prefix,
        }
    )
    meta.to_parquet(d / "meta.parquet", index=False)


def test_merge_dataset_versions(tmp_path: Path):
    root = tmp_path / "processed"
    _fake_dataset(root, "a", 3, seed=1, prefix="a")
    _fake_dataset(root, "b", 2, seed=2, prefix="b")
    out = merge_dataset_versions(
        root,
        ["a", "b"],
        "merged",
        overwrite=True,
        include_labels=False,
        source_tags=["nist", "sdbs"],
    )
    z = np.load(out / "spectra.npz", allow_pickle=True)
    meta = pd.read_parquet(out / "meta.parquet")
    assert z["X"].shape[0] == 5
    assert len(meta) == 5
    assert set(meta["source_dataset"]) == {"nist", "sdbs"}
    assert (out / "compounds.parquet").exists()
    assert (out / "manifest.json").exists()
    # collision: second a-copy skipped
    _fake_dataset(root, "a2", 3, seed=3, prefix="a")  # same ids as a
    out2 = merge_dataset_versions(
        root,
        ["a", "a2"],
        "merged_dup",
        overwrite=True,
        include_labels=False,
    )
    z2 = np.load(out2 / "spectra.npz", allow_pickle=True)
    assert z2["X"].shape[0] == 3
