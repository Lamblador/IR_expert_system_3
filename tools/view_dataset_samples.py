from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Просмотр спектров и структур (SMILES) из собранного датасета.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/processed/dataset_ai_project_v001"),
        help="Папка датасета (где лежат spectra.npz и meta.parquet).",
    )
    parser.add_argument(
        "--random-k",
        type=int,
        default=0,
        help="Сколько случайных записей показать.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed для случайной выборки.",
    )
    parser.add_argument(
        "--ids",
        nargs="*",
        default=None,
        help="Конкретные spectrum_id через пробел.",
    )
    parser.add_argument(
        "--indices",
        nargs="*",
        type=int,
        default=None,
        help="Конкретные индексы строк из meta.parquet через пробел.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/dataset_view"),
        help="Куда сохранять PNG.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Показывать окна matplotlib (вместо только сохранения).",
    )
    return parser.parse_args()


def load_dataset(dataset_dir: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray]:
    meta_path = dataset_dir / "meta.parquet"
    npz_path = dataset_dir / "spectra.npz"
    if not meta_path.exists() or not npz_path.exists():
        raise FileNotFoundError(f"Не найдены meta.parquet или spectra.npz в {dataset_dir}")

    meta = pd.read_parquet(meta_path).copy()
    npz = np.load(npz_path, allow_pickle=True)
    spectrum_ids = np.array(npz["spectrum_id"], dtype=object)
    X = np.asarray(npz["X"], dtype=np.float32)
    wavenumbers = np.asarray(npz["wavenumbers"], dtype=np.float32)

    id_to_idx = {str(sid): i for i, sid in enumerate(spectrum_ids)}
    if "spectrum_id" not in meta.columns:
        raise ValueError("В meta.parquet нет колонки spectrum_id")

    meta["spec_idx"] = meta["spectrum_id"].astype(str).map(id_to_idx)
    meta = meta[meta["spec_idx"].notna()].copy()
    meta["spec_idx"] = meta["spec_idx"].astype(int)

    return meta, {"X": X}, wavenumbers


def select_rows(meta: pd.DataFrame, random_k: int, seed: int, ids: list[str] | None, indices: list[int] | None) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []

    if ids:
        ids_set = {x.strip() for x in ids if x and x.strip()}
        if ids_set:
            selected.append(meta[meta["spectrum_id"].astype(str).isin(ids_set)])

    if indices:
        valid_indices = [i for i in indices if 0 <= i < len(meta)]
        if valid_indices:
            selected.append(meta.iloc[valid_indices])

    if random_k and random_k > 0:
        k = min(int(random_k), len(meta))
        selected.append(meta.sample(n=k, random_state=int(seed)))

    if not selected:
        raise ValueError("Ничего не выбрано: укажите --random-k или --ids или --indices")

    out = pd.concat(selected, axis=0).drop_duplicates(subset=["spectrum_id"]).reset_index(drop=True)
    if out.empty:
        raise ValueError("После фильтрации не осталось записей")
    return out


def draw_structure(smiles: str | None):
    if not smiles or not isinstance(smiles, str) or not smiles.strip():
        return np.ones((300, 300, 3), dtype=np.uint8) * 255
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.ones((300, 300, 3), dtype=np.uint8) * 255
    return Draw.MolToImage(mol, size=(380, 280))


def plot_entry(row: pd.Series, X: np.ndarray, wavenumbers: np.ndarray, out_dir: Path, show: bool) -> Path:
    spec_idx = int(row["spec_idx"])
    y = X[spec_idx]
    sid = str(row["spectrum_id"])
    title = str(row.get("title", ""))
    smiles = row.get("smiles")

    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.plot(wavenumbers, y, color="black", lw=1.0)
    ax1.set_xlim(float(np.max(wavenumbers)), float(np.min(wavenumbers)))
    ax1.set_xlabel(r"Wavenumber (cm$^{-1}$)")
    ax1.set_ylabel("Normalized absorbance-like")
    ax1.set_title(f"{title or sid}")
    ax1.grid(alpha=0.25)

    struct_img = draw_structure(smiles if isinstance(smiles, str) else None)
    ax2.imshow(struct_img)
    ax2.axis("off")
    ax2.set_title("Structure from SMILES")

    fig.suptitle(f"spectrum_id={sid}", fontsize=10)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{sid}.png"
    fig.savefig(out_path, dpi=160)
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    meta, arrays, wavenumbers = load_dataset(args.dataset_dir)
    picked = select_rows(meta, args.random_k, args.seed, args.ids, args.indices)

    print(f"Выбрано записей: {len(picked)}")
    print(f"Датасет: {args.dataset_dir}")
    for _, row in picked.iterrows():
        out_path = plot_entry(row, arrays["X"], wavenumbers, args.out_dir, args.show)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
