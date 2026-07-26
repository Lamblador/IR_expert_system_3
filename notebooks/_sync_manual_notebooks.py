"""One-shot: sync colab_06 + colab_launcher to dual-workflow header."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _make_colab_notebooks import code, md, nb, notebook_header  # noqa: E402

ROOT = Path(__file__).parent


def _cell_src(cell: dict) -> str:
    src = cell.get("source", "")
    if isinstance(src, list):
        return "".join(src)
    return str(src)


def main() -> None:
    path = ROOT / "colab_06_irresnet_train_drive.ipynb"
    old = json.loads(path.read_text(encoding="utf-8"))
    work = old["cells"][4:]

    header = notebook_header(
        "# Этап 6: IrResnet4 A/B на dataset_v002 (+ референс v003)\n\n"
        "**Цель:** два прогона на **dataset_v002** с протоколом "
        "`train_irresnet_original.yaml` (hidden=72, lr=1e-5, WRS, StepLR).\n\n"
        "| Run | label_schema | Смысл |\n"
        "|-----|--------------|-------|\n"
        "| `colab06_v002_smarts` | `structure_smarts` | SMARTS совпал |\n"
        "| `colab06_v002_structure` | `structure` | SMARTS и пик |\n\n"
        "Референс: `colab06_irresnet_dataset_v003` на v003.\n\n"
        "**Colab:** A + A2 + C. **Local:** B + C (`paths.local.yaml`; нужен v002 под processed_root).\n\n"
        "Документация: [`docs/NOTEBOOKS.md`](../docs/NOTEBOOKS.md).",
        include_drive=True,
        default_mode="auto",
        full_version="dataset_v003",
        fetch_hf_if_missing=False,
    )

    force_v002 = code(
        "# Для A/B эксперимента фиксируем dataset_v002 (перекрывает C при необходимости)\n"
        "from pathlib import Path\n"
        "from ir_pipeline.config_loader import load_yaml, resolve_paths\n\n"
        "TARGET_VERSION = 'dataset_v002'\n"
        "paths_cfg = load_yaml(PATHS_YAML)\n"
        "paths_cfg['dataset_version'] = TARGET_VERSION\n"
        "paths = resolve_paths(paths_cfg)\n"
        "DATASET_DIR = paths['processed_root'] / TARGET_VERSION\n"
        "BANDS_YAML = paths['bands_config']\n"
        "assert (DATASET_DIR / 'spectra.npz').is_file(), f'Нет spectra.npz: {DATASET_DIR}'\n"
        "assert (DATASET_DIR / 'labels_structure.parquet').is_file()\n"
        "assert (DATASET_DIR / 'labels_structure_smarts.parquet').is_file()\n"
        "if 'RUNS_DRIVE' not in globals() or RUNS_DRIVE is None:\n"
        "    RUNS_DRIVE = Path('runs')  # local fallback\n"
        "split_path = DATASET_DIR / 'split.json'\n"
        "print('dataset:', DATASET_DIR)\n"
        "print('PATHS_YAML:', PATHS_YAML)\n"
        "print('split:', 'ok' if split_path.is_file() else 'нет — выполните dataset-split ниже')\n"
    )

    patched_work: list[dict] = []
    for c in work:
        src = _cell_src(c)
        if "dataset-split" in src and ("paths.colab" in src or "--paths" in src):
            patched_work.append(
                code(
                    "import subprocess\n"
                    "subprocess.run([\n"
                    "    'ir-pipeline', 'dataset-split',\n"
                    "    '--paths', str(PATHS_YAML),\n"
                    "    '--dataset-version', 'dataset_v002',\n"
                    "    '--label-schema', 'structure_smarts',\n"
                    "], check=True)\n"
                    "split_path = DATASET_DIR / 'split.json'\n"
                    "print(\n"
                    "    split_path.read_text(encoding='utf-8')[:400]\n"
                    "    if split_path.is_file() else 'split.json не создан'\n"
                    ")\n"
                )
            )
            continue
        if "Saved to Drive" in src and "shutil.copytree" in src:
            patched_work.append(
                code(
                    "import shutil\n"
                    "from pathlib import Path\n\n"
                    "for run_suffix, run_dir in run_dirs.items():\n"
                    "    run_name = f'{run_suffix}_dataset_v002'\n"
                    "    dest_root = RUNS_DRIVE if RUNS_DRIVE is not None else Path('runs')\n"
                    "    dest = Path(dest_root) / run_name\n"
                    "    if dest.exists():\n"
                    "        shutil.rmtree(dest)\n"
                    "    shutil.copytree(run_dir, dest)\n"
                    "    shutil.make_archive(str(Path(dest_root) / run_name), 'zip', run_dir)\n"
                    "    print('Saved:', dest)\n"
                )
            )
            continue
        if "bands = paths['bands_config']" in src:
            src = src.replace("bands = paths['bands_config']", "bands = BANDS_YAML")
        if "bands_yaml=paths['bands_config']" in src:
            src = src.replace("bands_yaml=paths['bands_config']", "bands_yaml=BANDS_YAML")
        new_c = dict(c)
        new_c["source"] = [line + "\n" for line in src.splitlines()]
        if new_c["source"] and not src.endswith("\n"):
            new_c["source"][-1] = new_c["source"][-1].rstrip("\n")
        patched_work.append(new_c)

    out = nb(header + [md("### Фиксация dataset_v002 для A/B"), force_v002] + patched_work)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", path, "cells", len(out["cells"]))

    launcher = nb(
        [
            md(
                "# IR Pipeline — навигация (Local + Colab)\n\n"
                "## Выбор среды\n\n"
                "| Среда | В каждом ноутбуке |\n"
                "|-------|-------------------|\n"
                "| **Google Colab** | ячейка **A** (+ **A2** для Drive/full); **B** не запускать |\n"
                "| **Локальный Jupyter** | ячейка **B**; **A/A2** не запускать |\n\n"
                "Затем общая ячейка **C. Пути и данные**. Подробнее: "
                "[`docs/NOTEBOOKS.md`](../docs/NOTEBOOKS.md), "
                "[`docs/NOTEBOOK_API.md`](../docs/NOTEBOOK_API.md).\n\n"
                "| № | Ноутбук | Содержание |\n"
                "|---|---------|------------|\n"
                "| 0 | [`colab_00_setup.ipynb`](colab_00_setup.ipynb) | окружение + проверка датасета |\n"
                "| 1 | [`colab_01_dataset.ipynb`](colab_01_dataset.ipynb) | превью spectrum/structure |\n"
                "| 2 | [`colab_02_baseline_rf.ipynb`](colab_02_baseline_rf.ipynb) | RandomForest + MAE |\n"
                "| 3 | [`colab_03_train_irresnet4.ipynb`](colab_03_train_irresnet4.ipynb) | IrResnet4 |\n"
                "| 4 | [`colab_04_gradcam.ipynb`](colab_04_gradcam.ipynb) | Grad-CAM |\n"
                "| 5 | [`colab_05_irresnet_experiments.ipynb`](colab_05_irresnet_experiments.ipynb) | E1–E4 сравнение |\n"
                "| 6 | [`colab_06_irresnet_train_drive.ipynb`](colab_06_irresnet_train_drive.ipynb) | A/B v002 + original protocol |\n"
                "| 7 | [`colab_07_kan_compare.ipynb`](colab_07_kan_compare.ipynb) | KAN vs CNN M0/M1/M2 |\n\n"
                "Документация: [`docs/PIPELINE.md`](../docs/PIPELINE.md), "
                "[`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md).\n\n"
                "CLI smoke:\n\n"
                "```bash\n"
                "ir-pipeline run profile smoke --paths configs/paths.huggingface.yaml\n"
                "```\n"
            )
        ]
    )
    lp = ROOT / "colab_launcher.ipynb"
    lp.write_text(json.dumps(launcher, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", lp)


if __name__ == "__main__":
    main()
