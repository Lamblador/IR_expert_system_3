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


def mount_google_drive_cell() -> dict:
    return code(
        "from google.colab import drive\n"
        "drive.mount('/content/drive')\n"
        "from pathlib import Path\n"
        "IR_DATA = Path('/content/drive/MyDrive/ir_data')\n"
        "print('IR_DATA exists:', IR_DATA.exists(), IR_DATA)\n"
    )


def md_cnn_hyperparameters() -> dict:
    return md(
        "## Гиперпараметры обучения (CNN / IrResnet)\n\n"
        "| Параметр | По умолчанию (Colab) | Файл / как поменять |\n"
        "|----------|----------------------|---------------------|\n"
        "| **Эпохи** | `torch_epochs: 30` | `configs/train_irresnet_colab.yaml` |\n"
        "| **Learning rate** | `torch_lr: 0.001` | тот же yaml |\n"
        "| **Batch size** | `torch_batch_size: 32` | тот же yaml |\n"
        "| **Оптимизатор** | `torch_optimizer: adamw` | `adamw` \\| `adam` \\| `sgd` |\n"
        "| **Loss (IrResnet)** | `torch_loss: bce_with_logits` | multi-label BCE с logits |\n"
        "| **Loss (torch-train 1D CNN)** | `smooth_l1` | в `configs/train_torch_colab.yaml`: `smooth_l1` или `mse` |\n"
        "| **Размер скрытого слоя** | `ir_hidden_size: 34` | только IrResnet |\n"
        "| **Live-графики** | `live_training_plot: true` | в Colab: clear + график каждую эпоху; лог — последние 5 значений |\n\n"
        "В ячейке обучения ниже используется `--config configs/train_irresnet_colab.yaml`. "
        "Скопируйте yaml, измените числа, сохраните и укажите свой путь в `--config`.\n"
    )


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


def md_manual_dataset_upload() -> dict:
    return md(
        "## Датасет: загрузка вручную\n\n"
        "1. **Files → Upload** в Colab: `dataset_v001.zip` / `dataset_mini.zip` в `/content` "
        "(или положите архив на Google Drive).\n"
        "2. Выполните ячейку распаковки ниже — ожидается `data/processed/<версия>/spectra.npz`.\n"
        "3. Если архива нет — следующая ячейка скачает мини-датасет с Hugging Face.\n"
    )


def extract_manual_datasets_cell(
    versions: tuple[str, ...] = ("dataset_mini", "dataset_v001"),
) -> dict:
    versions_literal = repr(versions)
    return code(
        "from pathlib import Path\n"
        "import zipfile\n\n"
        f"DATASET_VERSIONS = {versions_literal}\n"
        "SEARCH_ROOTS = [\n"
        "    Path('/content'),\n"
        "    Path('/content/IR_expert_system_3'),\n"
        "    Path('/content/drive/MyDrive'),\n"
        "    Path('/content/drive/MyDrive/ir_data'),\n"
        "    Path('.'),\n"
        "]\n"
        "try:\n"
        "    SEARCH_ROOTS.insert(0, IR_DATA)\n"
        "except NameError:\n"
        "    pass\n"
        "DEST = Path('data/processed')\n"
        "DEST.mkdir(parents=True, exist_ok=True)\n\n"
        "def _dataset_ready(name: str) -> bool:\n"
        "    return (DEST / name / 'spectra.npz').is_file()\n\n"
        "def _find_zip_archives() -> list[Path]:\n"
        "    found: list[Path] = []\n"
        "    seen: set[str] = set()\n"
        "    for root in SEARCH_ROOTS:\n"
        "        if not root.exists():\n"
        "            continue\n"
        "        for p in root.rglob('*.zip'):\n"
        "            key = str(p.resolve())\n"
        "            if key in seen:\n"
        "                continue\n"
        "            low = p.name.lower()\n"
        "            if any(v in low for v in DATASET_VERSIONS):\n"
        "                seen.add(key)\n"
        "                found.append(p)\n"
        "    return sorted(found, key=lambda x: x.stat().st_mtime, reverse=True)\n\n"
        "archives = _find_zip_archives()\n"
        "print('Найденные zip с датасетом:')\n"
        "if archives:\n"
        "    for p in archives[:15]:\n"
        "        print(f'  {p} ({p.stat().st_size / 1e6:.1f} MB)')\n"
        "else:\n"
        "    print('  (нет — загрузите через Files → Upload)')\n\n"
        "for version in DATASET_VERSIONS:\n"
        "    if _dataset_ready(version):\n"
        "        print(f'OK: {DEST / version} уже распакован')\n"
        "        continue\n"
        "    matched = [p for p in archives if version in p.name.lower()]\n"
        "    if not matched:\n"
        "        print(f'Пропуск {version}: zip не найден')\n"
        "        continue\n"
        "    zp = matched[0]\n"
        "    print(f'Распаковка {zp.name} → {DEST}')\n"
        "    with zipfile.ZipFile(zp) as zf:\n"
        "        zf.extractall(DEST)\n"
        "    if _dataset_ready(version):\n"
        "        print(f'  → готово: {DEST / version / \"spectra.npz\"}')\n"
        "    else:\n"
        "        print(\n"
        "            f'  WARNING: после распаковки нет {DEST / version / \"spectra.npz\"}. '\n"
        "            'Проверьте структуру zip (внутри должна быть папка {version}/).'\n"
        "        )\n"
    )


def ensure_data_cell(
    dataset_version: str = "dataset_mini",
    hf_zip: str | None = None,
) -> dict:
    hf_zip = hf_zip or f"{dataset_version}.zip"
    return code(
        "from pathlib import Path\n\n"
        f"DATASET_DIR = Path('data/processed/{dataset_version}')\n"
        "if DATASET_DIR.joinpath('spectra.npz').is_file():\n"
        "    print(f'OK: {DATASET_DIR}')\n"
        "else:\n"
        f"    print('{dataset_version} not found → fetching from HF...')\n"
        f"    !ir-pipeline fetch-data --filename {hf_zip} --extract-to data/processed\n"
    )


def md_download_run() -> dict:
    return md(
        "## Сохранить обученную модель на локальный ПК\n\n"
        "Выполните ячейку ниже — браузер скачает zip каталога run "
        "(`models.joblib`, `metrics.json`, `irresnet_bundle.pt` и т.д.). "
        "На Windows распакуйте в `runs/<имя>/` и укажите `--run-dir`.\n"
    )


def download_run_zip_cell(run_dir: str, zip_name: str | None = None) -> dict:
    zip_stem = zip_name or run_dir.replace("\\", "/").strip("/").replace("/", "_")
    return code(
        "from pathlib import Path\n"
        "import shutil\n"
        "from google.colab import files\n\n"
        f"RUN_DIR = Path('{run_dir}')\n"
        "if not RUN_DIR.is_dir():\n"
        "    raise FileNotFoundError(\n"
        "        f'Нет {RUN_DIR} — сначала выполните ячейку обучения.'\n"
        "    )\n\n"
        "artifacts = [p for p in RUN_DIR.iterdir() if p.is_file()]\n"
        "if not artifacts:\n"
        "    raise FileNotFoundError(f'{RUN_DIR} пуст — нечего архивировать.')\n"
        "print('Файлы:', [p.name for p in sorted(artifacts)])\n\n"
        f"zip_path = Path('/content/{zip_stem}.zip')\n"
        "if zip_path.exists():\n"
        "    zip_path.unlink()\n"
        "shutil.make_archive(str(zip_path.with_suffix('')), 'zip', RUN_DIR)\n"
        "size_mb = zip_path.stat().st_size / 1e6\n"
        "print(f'Архив: {zip_path} ({size_mb:.2f} MB)')\n"
        "files.download(str(zip_path))\n"
        "print('Скачивание запущено.')\n"
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
            "Клонирует репозиторий и ставит зависимости."
        ),
        bootstrap_cell(),
        mount_google_drive_cell(),
        md_manual_dataset_upload(),
        extract_manual_datasets_cell(),
        ensure_data_cell(),
    ],
    "colab_01_dataset.ipynb": [
        md("# Этап 1: датасет и превью (автономный)\n\nSetup + HF fetch + графики spectrum/structure labels."),
        bootstrap_cell(),
        mount_google_drive_cell(),
        md_manual_dataset_upload(),
        extract_manual_datasets_cell(),
        ensure_data_cell(),
        code(
            "from pathlib import Path\nfrom IPython.display import Image, display\n"
            "from ir_pipeline.dataset_preview import plot_dataset_preview\n\n"
            "DATASET_DIR = Path('data/processed/dataset_mini')\n"
            "plot_dataset_preview(DATASET_DIR, Path('runs/colab_preview/plots'), Path('configs/bands_reference.yaml'))\n"
            "plots = sorted(Path('runs').rglob('preview_spectrum_0.png'))\n"
            "if plots:\n"
            "    display(Image(filename=str(plots[-1]), width=900))\n"
        ),
    ],
    "colab_02_baseline_rf.ipynb": [
        md("# Этап 2: baseline RandomForest (автономный)\n\nSetup + dataset + RF + графики MAE."),
        bootstrap_cell(),
        mount_google_drive_cell(),
        md_manual_dataset_upload(),
        extract_manual_datasets_cell(),
        ensure_data_cell(),
        find_jcamp_or_zip_cell(),
        code(
            "from pathlib import Path\nfrom IPython.display import Image, display\n\n"
            "RUN_DIR = Path('runs/colab_pipeline_rf/rf_run')\n"
            "!ir-pipeline train --paths configs/paths.huggingface.yaml --dataset-version dataset_mini "
            "--mode spectrum --config configs/train_mini.yaml --run-dir {RUN_DIR}\n"
            "!ir-pipeline plot-train-metrics --run-dir {RUN_DIR}\n\n"
            "for pat in ['metrics_per_band_mae.png', 'metrics_by_group_mae.png']:\n"
            "    hits = sorted(Path('runs').rglob(pat))\n"
            "    if hits:\n"
            "        display(Image(filename=str(hits[-1]), width=900))\n"
        ),
        md_download_run(),
        download_run_zip_cell("runs/colab_pipeline_rf/rf_run", "rf_run_colab"),
    ],
    "colab_03_train_irresnet4.ipynb": [
        md("# Этап 3: IrResnet4 multi-label (автономный)\n\n3-канальный вход 400–4000 см⁻¹ + контекст ATR/gas/solution."),
        bootstrap_cell(),
        mount_google_drive_cell(),
        md_manual_dataset_upload(),
        extract_manual_datasets_cell(),
        ensure_data_cell(),
        md_cnn_hyperparameters(),
        code(
            "from pathlib import Path\nfrom IPython.display import Image, display\n\n"
            "RUN_DIR = Path('runs/colab_pipeline_irresnet/irresnet_run')\n"
            "!ir-pipeline irresnet-train --paths configs/paths.huggingface.yaml "
            "--dataset-version dataset_mini --config configs/train_irresnet_colab.yaml --run-dir {RUN_DIR}\n\n"
            "hits = sorted(Path('runs').rglob('irresnet_training_curve.png'))\n"
            "if hits:\n"
            "    display(Image(filename=str(hits[-1]), width=900))\n"
        ),
        md_download_run(),
        download_run_zip_cell("runs/colab_pipeline_irresnet/irresnet_run", "irresnet_run_colab"),
    ],
    "colab_04_gradcam.ipynb": [
        md(
            "# Этап 4: Grad-CAM вручную (автономный)\n\n"
            "Наложение карт важности на спектр. Можно задать индексы спектров и классов."
        ),
        bootstrap_cell(),
        mount_google_drive_cell(),
        md_manual_dataset_upload(),
        extract_manual_datasets_cell(),
        ensure_data_cell(),
        code(
            "from pathlib import Path\nfrom IPython.display import Image, display\n\n"
            "PIPELINE_RUN = Path('runs/colab_pipeline_gradcam')\n"
            "bundles = sorted(Path('runs').rglob('irresnet_bundle.pt'))\n"
            "if not bundles:\n"
            "    print('Нет обученной модели — тренируем...')\n"
            "    !ir-pipeline irresnet-train --paths configs/paths.huggingface.yaml "
            "--dataset-version dataset_mini --config configs/train_irresnet.yaml "
            "--run-dir {PIPELINE_RUN / 'irresnet_run'}\n"
            "    bundles = sorted(Path('runs').rglob('irresnet_bundle.pt'))\n\n"
            "IR_RUN = bundles[-1].parent\n"
            "OUT = Path('reports/colab_gradcam')\n"
            "# Авто: первые 3 спектра. Ручной режим — раскомментируйте:\n"
            "# !ir-pipeline gradcam-examples --paths configs/paths.huggingface.yaml --run-dir {IR_RUN} "
            "--output-dir {OUT} --spectrum-indices 0,7,15 --class-indices 3,12\n"
            "!ir-pipeline gradcam-examples --paths configs/paths.huggingface.yaml "
            "--run-dir {IR_RUN} --output-dir {OUT} --n-examples 3\n\n"
            "for p in sorted(OUT.glob('gradcam_*.png'))[:3]:\n"
            "    display(Image(filename=str(p), width=900))\n"
        ),
        md_download_run(),
        code(
            "from pathlib import Path\n"
            "import shutil\n"
            "from google.colab import files\n\n"
            "bundles = sorted(Path('runs').rglob('irresnet_bundle.pt'))\n"
            "if not bundles:\n"
            "    raise FileNotFoundError('Нет irresnet_bundle.pt — сначала обучите модель.')\n"
            "RUN_DIR = bundles[-1].parent\n"
            "print('Run dir:', RUN_DIR)\n\n"
            "zip_path = Path('/content/irresnet_gradcam_run_colab.zip')\n"
            "if zip_path.exists():\n"
            "    zip_path.unlink()\n"
            "shutil.make_archive(str(zip_path.with_suffix('')), 'zip', RUN_DIR)\n"
            "print(f'Архив: {zip_path} ({zip_path.stat().st_size / 1e6:.2f} MB)')\n"
            "files.download(str(zip_path))\n"
        ),
    ],
}


if __name__ == "__main__":
    for name, cells in NOTEBOOKS.items():
        path = ROOT / name
        path.write_text(json.dumps(nb(cells), ensure_ascii=False, indent=1), encoding="utf-8")
        print("wrote", path)
