"""Генератор Colab-ноутбуков (каждый ноутбук автономный)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parent


def nb(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
            "colab": {"provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source.splitlines(keepends=True),
        "outputs": [],
        "execution_count": None,
    }


def bootstrap_cell(extra: str = "") -> dict:
    src = (
        "import subprocess\n"
        "from pathlib import Path\n\n"
        "REPO_URL = \"https://github.com/Lamblador/IR_expert_system_3.git\"  # при необходимости замените\n"
        "REPO_DIR = Path(\"IR_expert_system_3\")\n\n"
        "if not REPO_DIR.is_dir():\n"
        "    subprocess.run([\"git\", \"clone\", REPO_URL, str(REPO_DIR)], check=True)\n\n"
        "%cd IR_expert_system_3\n"
        "!pip install -q -e \".[torch]\"\n"
        "!ir-pipeline --help\n"
        "import subprocess\n"
        "help_txt = subprocess.check_output(['ir-pipeline', '--help'], text=True)\n"
        "if ' run ' not in help_txt:\n"
        "    print('WARNING: команда `run` отсутствует. Ноутбук использует fallback без run-stage.')\n"
    )
    if extra:
        src += "\n" + extra + "\n"
    return code(src)


def ensure_data_cell() -> dict:
    return code(
        "from pathlib import Path\n\n"
        "DATASET_DIR = Path('data/processed/dataset_mini')\n"
        "if not DATASET_DIR.exists():\n"
        "    print('dataset_mini not found → fetching from HF...')\n"
        "    !ir-pipeline fetch-data --filename dataset_mini.zip --extract-to data/processed\n"
        "else:\n"
        "    print(f'{DATASET_DIR} already exists')\n"
    )


def find_jcamp_or_zip_cell() -> dict:
    return code(
        "from pathlib import Path\n"
        "import zipfile, shutil\n\n"
        "SEARCH_ROOTS = [Path('/content'), Path('/content/IR_expert_system_3'), Path('/content/drive/MyDrive')]\n"
        "candidates = []\n"
        "for root in SEARCH_ROOTS:\n"
        "    if not root.exists():\n"
        "        continue\n"
        "    for p in root.rglob('*'):\n"
        "        name = p.name.lower()\n"
        "        if p.is_dir() and name == 'downloaded_jcamp':\n"
        "            candidates.append(('dir', p))\n"
        "        if p.is_file() and ('downloaded_jcamp' in name and name.endswith('.zip')):\n"
        "            candidates.append(('zip', p))\n\n"
        "print('Found candidates:')\n"
        "for k, p in candidates[:30]:\n"
        "    print(k, p)\n\n"
        "target = Path('/content/IR_expert_system_3/downloaded_jcamp')\n"
        "target.parent.mkdir(parents=True, exist_ok=True)\n"
        "if not target.exists():\n"
        "    for kind, p in candidates:\n"
        "        if kind == 'dir':\n"
        "            print('Copying directory to', target)\n"
        "            shutil.copytree(p, target, dirs_exist_ok=True)\n"
        "            break\n"
        "        if kind == 'zip':\n"
        "            print('Extracting zip to', target)\n"
        "            target.mkdir(parents=True, exist_ok=True)\n"
        "            with zipfile.ZipFile(p) as zf:\n"
        "                zf.extractall(target)\n"
        "            break\n"
        "print('downloaded_jcamp exists:', target.exists())\n"
    )


NOTEBOOKS = {
    "colab_00_setup.ipynb": [
        md(
            "# Этап 0: установка (автономный)\n\n"
            "Этот ноутбук можно запускать отдельно на новом Colab runtime.\n\n"
            "Клонирует репозиторий и ставит зависимости."
        ),
        bootstrap_cell(),
    ],
    "colab_01_dataset.ipynb": [
        md(
            "# Этап 1: датасет (автономный)\n\n"
            "Сам делает setup + при необходимости скачивает `dataset_mini`."
        ),
        bootstrap_cell(),
        ensure_data_cell(),
        code(
            "from pathlib import Path\nfrom IPython.display import Image, display\n"
            "from ir_pipeline.dataset_telegram import build_telegram_arrays_from_npz, plot_dataset_preview, save_telegram_npz\n\n"
            "DATASET_DIR = Path('data/processed/dataset_mini')\n"
            "if not (DATASET_DIR / 'telegram_arrays.npz').exists():\n"
            "    X_bot, ids = build_telegram_arrays_from_npz(DATASET_DIR)\n"
            "    save_telegram_npz(DATASET_DIR, X_bot, ids)\n"
            "plot_dataset_preview(DATASET_DIR, Path('runs/colab_preview/plots'), Path('configs/bands_reference.yaml'))\n"
            "plots = sorted(Path('runs').rglob('preview_spectrum_0.png'))\n"
            "if plots:\n"
            "    display(Image(filename=str(plots[-1]), width=900))\n"
            "else:\n"
            "    print('Нет preview PNG — проверьте error_log.txt в stage_03')\n"
        ),
    ],
    "colab_02_baseline_rf.ipynb": [
        md(
            "# Этап 2: baseline RandomForest (автономный)\n\n"
            "Setup + auto-fetch dataset + обучение RF + графики MAE."
        ),
        bootstrap_cell(),
        ensure_data_cell(),
        find_jcamp_or_zip_cell(),
        code(
            "from pathlib import Path\nfrom IPython.display import Image, display\n\n"
            "RUN_DIR = Path('runs/colab_pipeline_rf/rf_run')\n"
            "# GPU (опционально):\n"
            "# !pip install -q -e \".[cuml]\"\n"
            "# import os; os.environ['IR_RF_BACKEND'] = 'cuml'\n\n"
            "!ir-pipeline train --paths configs/paths.huggingface.yaml --dataset-version dataset_mini --mode spectrum --config configs/train_mini.yaml --run-dir {RUN_DIR}\n"
            "!ir-pipeline plot-train-metrics --run-dir {RUN_DIR}\n\n"
            "for pat in ['metrics_per_band_mae.png', 'metrics_by_group_mae.png']:\n"
            "    hits = sorted(Path('runs').rglob(pat))\n"
            "    if hits:\n"
            "        print(hits[-1])\n"
            "        display(Image(filename=str(hits[-1]), width=900))\n"
        ),
    ],
    "colab_03_train_irresnet4.ipynb": [
        md(
            "# Этап 3: IrResnet4 (автономный)\n\n"
            "Setup + auto-fetch dataset + обучение классификационной модели."
        ),
        bootstrap_cell(),
        ensure_data_cell(),
        code(
            "from pathlib import Path\nfrom IPython.display import Image, display\n\n"
            "RUN_DIR = Path('runs/colab_pipeline_irresnet/irresnet_run')\n"
            "!ir-pipeline irresnet-train --paths configs/paths.huggingface.yaml --dataset-version dataset_mini --config configs/train_irresnet.yaml --run-dir {RUN_DIR}\n\n"
            "hits = sorted(Path('runs').rglob('irresnet_training_curve.png'))\n"
            "if hits:\n"
            "    display(Image(filename=str(hits[-1]), width=900))\n"
        ),
    ],
    "colab_04_cam_examples.ipynb": [
        md(
            "# Этап 4: Grad-CAM (автономный)\n\n"
            "Setup + auto-fetch dataset + при отсутствии модели сначала тренирует IrResnet4."
        ),
        bootstrap_cell(),
        ensure_data_cell(),
        code(
            "from pathlib import Path\nfrom IPython.display import Image, display\n\n"
            "PIPELINE_RUN = Path('runs/colab_pipeline_cam')\n"
            "bundles = sorted(Path('runs').rglob('irresnet_bundle.pt'))\n"
            "if not bundles:\n"
            "    print('No irresnet bundle found, training one...')\n"
            "    !ir-pipeline irresnet-train --paths configs/paths.huggingface.yaml --dataset-version dataset_mini --config configs/train_irresnet.yaml --run-dir {PIPELINE_RUN / 'irresnet_run'}\n"
            "    bundles = sorted(Path('runs').rglob('irresnet_bundle.pt'))\n\n"
            "IR_RUN = bundles[-1].parent\n"
            "print('Using', IR_RUN)\n"
            "!ir-pipeline cam-examples --paths configs/paths.huggingface.yaml --run-dir {IR_RUN} --output-dir reports/colab_cam\n\n"
            "for p in sorted(Path('reports/colab_cam').glob('cam_example_*.png'))[:3]:\n"
            "    display(Image(filename=str(p), width=900))\n"
        ),
    ],
    "colab_05_export_telegram.ipynb": [
        md(
            "# Этап 5: экспорт в Telegram-бот (автономный)\n\n"
            "Setup + auto-fetch dataset + при отсутствии модели сначала тренирует IrResnet4."
        ),
        bootstrap_cell(),
        ensure_data_cell(),
        code(
            "from pathlib import Path\n\n"
            "bundles = sorted(Path('runs').rglob('irresnet_bundle.pt'))\n"
            "if not bundles:\n"
            "    print('No irresnet bundle found, training one...')\n"
            "    !ir-pipeline irresnet-train --paths configs/paths.huggingface.yaml --dataset-version dataset_mini --config configs/train_irresnet.yaml --run-dir runs/colab_pipeline_export/irresnet_run\n"
            "    bundles = sorted(Path('runs').rglob('irresnet_bundle.pt'))\n\n"
            "IR_RUN = bundles[-1].parent\n"
            "TARGET = Path('/content/drive/MyDrive/ftir_bot_models')  # поправьте под свой путь\n"
            "TARGET.mkdir(parents=True, exist_ok=True)\n"
            "!ir-pipeline export-telegram --run-dir {IR_RUN} --target-dir {TARGET}\n"
            "print('Files:', list(TARGET.rglob('*_model_param')))\n"
        ),
    ],
}


if __name__ == "__main__":
    for name, cells in NOTEBOOKS.items():
        path = ROOT / name
        path.write_text(json.dumps(nb(cells), ensure_ascii=False, indent=1), encoding="utf-8")
        print("wrote", path)
