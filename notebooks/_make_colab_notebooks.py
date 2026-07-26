"""Генератор ноутбуков dual-workflow (Local Jupyter + Google Colab).

Источник истины для colab_00…05 и colab_07. Запуск:
  python notebooks/_make_colab_notebooks.py
"""
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


# ---------------------------------------------------------------------------
# Dual-workflow header (A Colab / B Local / C paths+data)
# ---------------------------------------------------------------------------

def md_env_choice() -> dict:
    return md(
        "## Выбор среды (выполните ОДНУ ячейку)\n\n"
        "| Среда | Запустить | Пропустить |\n"
        "|-------|-----------|------------|\n"
        "| **Google Colab** | **A. Colab** (+ при полном датасете **A2. Drive**) | **B. Local** |\n"
        "| **Локальный Jupyter** | **B. Local** | **A** и **A2** (включая `drive.mount`) |\n\n"
        "После A или B выполните **C. Пути и данные**.\n"
        "Подробнее: [`docs/NOTEBOOKS.md`](../docs/NOTEBOOKS.md).\n"
    )


def cell_env_colab() -> dict:
    """A. Colab: clone в /content, pip install. Не запускать локально."""
    return code(
        "# === A. Colab: окружение ===\n"
        "# Локально эту ячейку НЕ запускайте (см. B. Local).\n"
        "import os\n"
        "import subprocess\n"
        "import sys\n"
        "from pathlib import Path\n\n"
        "REPO_URL = 'https://github.com/Lamblador/IR_expert_system_3.git'\n"
        "REPO_BRANCH = 'colab-v1'\n"
        "REPO_DIR = Path('/content/IR_expert_system_3')\n\n"
        "def _run_git(cmd, cwd=None):\n"
        "    print('git', ' '.join(cmd))\n"
        "    subprocess.run(cmd, cwd=cwd, check=True)\n\n"
        "if (REPO_DIR / '.git').is_dir():\n"
        "    _run_git(['git', 'fetch', 'origin', REPO_BRANCH], cwd=REPO_DIR)\n"
        "    _run_git(['git', 'checkout', REPO_BRANCH], cwd=REPO_DIR)\n"
        "    _run_git(['git', 'pull', '--ff-only', 'origin', REPO_BRANCH], cwd=REPO_DIR)\n"
        "else:\n"
        "    if REPO_DIR.exists():\n"
        "        raise RuntimeError(f'{REPO_DIR} существует, но это не git-репозиторий')\n"
        "    _run_git([\n"
        "        'git', 'clone', '-b', REPO_BRANCH, '--single-branch',\n"
        "        REPO_URL, str(REPO_DIR),\n"
        "    ])\n\n"
        "ROOT = REPO_DIR.resolve()\n"
        "os.chdir(ROOT)\n"
        "try:\n"
        "    from IPython import get_ipython\n"
        "    get_ipython().run_line_magic('cd', str(ROOT))\n"
        "except Exception:\n"
        "    pass\n"
        "rev = subprocess.check_output(\n"
        "    ['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, text=True\n"
        ").strip()\n"
        "print(f'ROOT: {ROOT} @ {REPO_BRANCH} ({rev})')\n"
        "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', '.[torch]'], check=True)\n"
        "print('pip install OK')\n"
        "IR_ENV = 'colab'\n"
    )


def cell_env_colab_drive() -> dict:
    """A2. Mount Drive — только Colab full. Не запускать локально / smoke."""
    return code(
        "# === A2. Colab Drive (full dataset) ===\n"
        "# Нужен только для полного датасета на Google Drive.\n"
        "# Для HF smoke (dataset_mini) эту ячейку ПРОПУСТИТЕ.\n"
        "# Локально НЕ запускайте.\n"
        "from pathlib import Path\n"
        "from google.colab import drive\n\n"
        "drive.mount('/content/drive')\n"
        "IR_DATA = Path('/content/drive/MyDrive/ir_data')\n"
        "RUNS_DRIVE = Path('/content/drive/MyDrive/ir_expert_system_3/runs')\n"
        "RUNS_DRIVE.mkdir(parents=True, exist_ok=True)\n"
        "print('IR_DATA exists:', IR_DATA.exists(), IR_DATA)\n"
        "print('RUNS_DRIVE:', RUNS_DRIVE)\n"
    )


def cell_env_local() -> dict:
    """B. Local: walk-up к pyproject.toml, без checkout ветки."""
    return code(
        "# === B. Local: окружение ===\n"
        "# В Google Colab эту ячейку НЕ запускайте (см. A. Colab).\n"
        "import os\n"
        "import subprocess\n"
        "import sys\n"
        "from pathlib import Path\n\n"
        "def _find_repo_root(start: Path) -> Path:\n"
        "    \"\"\"Walk-up до каталога с pyproject.toml (фикс nested-clone из notebooks/).\"\"\"\n"
        "    cur = start.resolve()\n"
        "    for p in [cur, *cur.parents]:\n"
        "        if (p / 'pyproject.toml').is_file():\n"
        "            return p\n"
        "    raise FileNotFoundError(\n"
        "        'Не найден pyproject.toml выше cwd. '\n"
        "        'Откройте ноутбук из клона репозитория или cd в корень IR_expert_system_3.'\n"
        "    )\n\n"
        "ROOT = _find_repo_root(Path.cwd())\n"
        "os.chdir(ROOT)\n"
        "try:\n"
        "    from IPython import get_ipython\n"
        "    get_ipython().run_line_magic('cd', str(ROOT))\n"
        "except Exception:\n"
        "    pass\n"
        "print('ROOT (local, ветку не переключаем):', ROOT)\n\n"
        "FORCE_REINSTALL = False  # True — принудительно pip install -e .[torch]\n"
        "need_install = FORCE_REINSTALL\n"
        "if not need_install:\n"
        "    try:\n"
        "        import ir_pipeline  # noqa: F401\n"
        "    except ImportError:\n"
        "        need_install = True\n"
        "if need_install:\n"
        "    subprocess.run(\n"
        "        [sys.executable, '-m', 'pip', 'install', '-q', '-e', '.[torch]'],\n"
        "        check=True,\n"
        "    )\n"
        "    print('pip install OK')\n"
        "else:\n"
        "    print('ir_pipeline уже установлен — pip пропущен (FORCE_REINSTALL=True для переустановки)')\n"
        "IR_ENV = 'local'\n"
    )


def cell_paths_and_data(
    *,
    default_mode: str = "auto",
    smoke_version: str = "dataset_mini",
    full_version: str = "dataset_v003",
    fetch_hf_if_missing: bool = True,
) -> dict:
    """C. Пути: local / colab_smoke / colab_full → ROOT, PATHS_YAML, paths, DATASET_DIR, …"""
    if fetch_hf_if_missing:
        missing_block = (
            "    if DATA_MODE == 'colab_smoke':\n"
            "        print(f'{DATASET_DIR} нет → fetch HF {SMOKE_VERSION}.zip')\n"
            f"        !ir-pipeline fetch-data --filename {smoke_version}.zip --extract-to data/processed\n"
            "        if not spectra.is_file():\n"
            "            raise FileNotFoundError(f'После fetch нет {spectra}')\n"
            "    else:\n"
            "        raise FileNotFoundError(\n"
            "            f'Нет {spectra}. Проверьте paths yaml / Drive / локальные каталоги. '\n"
            "            f'DATA_MODE={DATA_MODE}, PATHS_YAML={PATHS_YAML}'\n"
            "        )\n"
        )
    else:
        missing_block = (
            "    raise FileNotFoundError(\n"
            "        f'Нет {spectra}. DATA_MODE={DATA_MODE}, PATHS_YAML={PATHS_YAML}'\n"
            "    )\n"
        )
    src = (
        "# === C. Пути и данные ===\n"
        "# Выполните после A или B. Контракт: ROOT, PATHS_YAML, paths, DATASET_DIR, BANDS_YAML, RUNS_DIR\n"
        "import os\n"
        "from pathlib import Path\n"
        "from ir_pipeline.config_loader import load_yaml, resolve_paths\n\n"
        f"SMOKE_VERSION = '{smoke_version}'\n"
        f"FULL_VERSION = '{full_version}'\n"
        "# Режим данных (если IR_ENV не задан — auto):\n"
        "#   'local'       — configs/paths.local.yaml\n"
        "#   'colab_smoke' — HF mini, paths.huggingface.yaml\n"
        "#   'colab_full'  — Drive, paths.colab.yaml\n"
        f"DATA_MODE = '{default_mode}'  # или 'local' | 'colab_smoke' | 'colab_full'\n\n"
        "if 'IR_ENV' not in globals():\n"
        "    IR_ENV = 'colab' if Path('/content').exists() else 'local'\n\n"
        "if DATA_MODE == 'auto':\n"
        "    if IR_ENV == 'local':\n"
        "        DATA_MODE = 'local'\n"
        "    elif 'IR_DATA' in globals() and Path(IR_DATA).exists():\n"
        "        DATA_MODE = 'colab_full'\n"
        "    else:\n"
        "        DATA_MODE = 'colab_smoke'\n\n"
        "if DATA_MODE == 'local':\n"
        "    PATHS_YAML = Path('configs/paths.local.yaml')\n"
        "    if not PATHS_YAML.is_file():\n"
        "        raise FileNotFoundError(\n"
        "            'Нет configs/paths.local.yaml — скопируйте configs/paths.local.example.yaml '\n"
        "            'и пропишите raw_jcamp_dir / processed_root.'\n"
        "        )\n"
        "elif DATA_MODE == 'colab_full':\n"
        "    if 'IR_DATA' not in globals():\n"
        "        raise RuntimeError('Colab full: сначала выполните A2 (Drive mount) → IR_DATA')\n"
        "    os.environ['IR_PROCESSED_ROOT'] = str(Path(IR_DATA) / 'processed')\n"
        "    PATHS_YAML = Path('configs/paths.colab.yaml')\n"
        "elif DATA_MODE == 'colab_smoke':\n"
        "    PATHS_YAML = Path('configs/paths.huggingface.yaml')\n"
        "else:\n"
        "    raise ValueError(f'Неизвестный DATA_MODE={DATA_MODE!r}')\n\n"
        "paths_cfg = load_yaml(PATHS_YAML)\n"
        "if DATA_MODE == 'colab_smoke':\n"
        "    paths_cfg['dataset_version'] = SMOKE_VERSION\n"
        "    paths_cfg.pop('dataset_profile', None)\n"
        "elif DATA_MODE == 'colab_full':\n"
        "    paths_cfg['dataset_version'] = paths_cfg.get('dataset_version') or FULL_VERSION\n"
        "# local: dataset_version / profile из yaml\n\n"
        "paths = resolve_paths(paths_cfg)\n"
        "DATASET_DIR = paths['processed_root'] / str(paths['dataset_version'])\n"
        "BANDS_YAML = paths['bands_config']\n"
        "RUNS_DIR = Path('runs')\n"
        "RUNS_DIR.mkdir(parents=True, exist_ok=True)\n\n"
        "spectra = DATASET_DIR / 'spectra.npz'\n"
        "if not spectra.is_file():\n"
        + missing_block
        + "print('DATA_MODE:', DATA_MODE)\n"
        "print('PATHS_YAML:', PATHS_YAML)\n"
        "print('DATASET_DIR:', DATASET_DIR)\n"
        "print('dataset_version:', paths['dataset_version'])\n"
        "print('raw_jcamp_dir:', paths['raw_jcamp_dir'])\n"
    )
    return code(src)


def notebook_header(
    title_md: str,
    *,
    include_drive: bool = True,
    default_mode: str = "auto",
    smoke_version: str = "dataset_mini",
    full_version: str = "dataset_v003",
    fetch_hf_if_missing: bool = True,
) -> list[dict]:
    """Стандартная шапка: title + env choice + A [+A2] + B + C."""
    cells = [md(title_md), md_env_choice(), cell_env_colab()]
    if include_drive:
        cells.append(md("### A2. Google Drive (только Colab full)\n\nПропустите для HF smoke и локально."))
        cells.append(cell_env_colab_drive())
    cells.extend(
        [
            md("### B. Локальный Jupyter\n\nПропустите в Colab."),
            cell_env_local(),
            md("### C. Пути и данные\n\nПосле A или B."),
            cell_paths_and_data(
                default_mode=default_mode,
                smoke_version=smoke_version,
                full_version=full_version,
                fetch_hf_if_missing=fetch_hf_if_missing,
            ),
        ]
    )
    return cells


def md_cnn_hyperparameters() -> dict:
    return md(
        "## Гиперпараметры обучения (CNN / IrResnet)\n\n"
        "| Параметр | По умолчанию (Colab) | Файл / как поменять |\n"
        "|----------|----------------------|---------------------|\n"
        "| **Эпохи** | `torch_epochs: 30` | `configs/train_irresnet_colab.yaml` |\n"
        "| **Learning rate** | `torch_lr: 0.001` | тот же yaml |\n"
        "| **Batch size** | `torch_batch_size: 32` | тот же yaml |\n"
        "| **Оптимизатор** | `torch_optimizer: adamw` | `adamw` \\| `adam` \\| `sgd` |\n"
        "| **Метки** | `structure` / `structure_smarts` / `spectrum` | `--label-schema` или kwarg |\n"
        "| **Loss (IrResnet)** | `torch_loss: bce_with_logits` | multi-label BCE |\n"
        "| **Hidden** | `ir_hidden_size: 34` | только IrResnet |\n"
        "| **Live-графики** | `live_training_plot: true` | Colab: clear + график |\n\n"
        "Ниже используется `configs/train_irresnet_colab.yaml`. "
        "Скопируйте yaml и укажите свой `--config`.\n"
    )


def md_manual_dataset_upload() -> dict:
    return md(
        "## (Опционально) Zip вручную в Colab\n\n"
        "1. **Files → Upload**: `dataset_mini.zip` / `dataset_v003.zip` в `/content` или Drive.\n"
        "2. Ячейка ниже ищет и распаковывает в `data/processed/`.\n"
        "3. Иначе ячейка **C** скачает mini с Hugging Face (режим `colab_smoke`).\n"
    )


def extract_manual_datasets_cell(
    versions: tuple[str, ...] = ("dataset_mini", "dataset_v003"),
) -> dict:
    versions_literal = repr(versions)
    return code(
        "# Опционально: распаковка zip (Colab). Локально обычно не нужна.\n"
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
        "if 'IR_DATA' in globals():\n"
        "    SEARCH_ROOTS.insert(0, Path(IR_DATA))\n"
        "DEST = Path('data/processed')\n"
        "DEST.mkdir(parents=True, exist_ok=True)\n\n"
        "def _dataset_ready(name: str) -> bool:\n"
        "    return (DEST / name / 'spectra.npz').is_file()\n\n"
        "def _find_zip_archives() -> list[Path]:\n"
        "    found, seen = [], set()\n"
        "    for root in SEARCH_ROOTS:\n"
        "        if not root.exists():\n"
        "            continue\n"
        "        for p in root.rglob('*.zip'):\n"
        "            key = str(p.resolve())\n"
        "            if key in seen:\n"
        "                continue\n"
        "            if any(v in p.name.lower() for v in DATASET_VERSIONS):\n"
        "                seen.add(key)\n"
        "                found.append(p)\n"
        "    return sorted(found, key=lambda x: x.stat().st_mtime, reverse=True)\n\n"
        "archives = _find_zip_archives()\n"
        "print('Найденные zip:')\n"
        "for p in archives[:15]:\n"
        "    print(f'  {p} ({p.stat().st_size / 1e6:.1f} MB)')\n"
        "if not archives:\n"
        "    print('  (нет)')\n\n"
        "for version in DATASET_VERSIONS:\n"
        "    if _dataset_ready(version):\n"
        "        print(f'OK: {DEST / version}')\n"
        "        continue\n"
        "    matched = [p for p in archives if version in p.name.lower()]\n"
        "    if not matched:\n"
        "        print(f'Пропуск {version}: zip не найден')\n"
        "        continue\n"
        "    zp = matched[0]\n"
        "    print(f'Распаковка {zp.name} → {DEST}')\n"
        "    with zipfile.ZipFile(zp) as zf:\n"
        "        zf.extractall(DEST)\n"
        "    print('  →', 'OK' if _dataset_ready(version) else 'WARNING: нет spectra.npz')\n"
    )


def md_download_run() -> dict:
    return md(
        "## Сохранить run (только Colab)\n\n"
        "Ячейка ниже скачает zip через браузер. **Локально пропустите** — артефакты уже в `runs/`.\n"
    )


def download_run_zip_cell(run_dir: str, zip_name: str | None = None) -> dict:
    zip_stem = zip_name or run_dir.replace("\\", "/").strip("/").replace("/", "_")
    return code(
        "# Только Colab. Локально пропустите.\n"
        "from pathlib import Path\n"
        "import shutil\n\n"
        "try:\n"
        "    from google.colab import files\n"
        "except ImportError:\n"
        f"    print('Не Colab — скачивание пропущено. Смотрите', Path('{run_dir}'))\n"
        "else:\n"
        f"    RUN_DIR = Path('{run_dir}')\n"
        "    if not RUN_DIR.is_dir():\n"
        "        raise FileNotFoundError(f'Нет {RUN_DIR} — сначала обучите модель.')\n"
        "    artifacts = [p for p in RUN_DIR.iterdir() if p.is_file()]\n"
        "    if not artifacts:\n"
        "        raise FileNotFoundError(f'{RUN_DIR} пуст')\n"
        "    print('Файлы:', [p.name for p in sorted(artifacts)])\n"
        f"    zip_path = Path('/content/{zip_stem}.zip')\n"
        "    if zip_path.exists():\n"
        "        zip_path.unlink()\n"
        "    shutil.make_archive(str(zip_path.with_suffix('')), 'zip', RUN_DIR)\n"
        "    print(f'Архив: {zip_path} ({zip_path.stat().st_size / 1e6:.2f} MB)')\n"
        "    files.download(str(zip_path))\n"
    )


# ---------------------------------------------------------------------------
# Notebook definitions
# ---------------------------------------------------------------------------

NOTEBOOKS = {
    "colab_00_setup.ipynb": [
        *notebook_header(
            "# Этап 0: установка\n\n"
            "**Цель:** окружение (Colab или локально) + проверка `ir-pipeline` и датасета.\n\n"
            "**Выход:** `ROOT`, `DATASET_DIR`, установленный пакет `ir_pipeline`.\n\n"
            "Документация: [`docs/NOTEBOOKS.md`](../docs/NOTEBOOKS.md).",
            include_drive=True,
            default_mode="auto",
        ),
        md_manual_dataset_upload(),
        extract_manual_datasets_cell(),
        code(
            "# Проверка CLI\n"
            "import subprocess\n"
            "help_txt = subprocess.check_output(['ir-pipeline', '--help'], text=True)\n"
            "print('ir-pipeline OK; run stage:', ' run ' in help_txt)\n"
            "print('DATASET_DIR:', DATASET_DIR)\n"
            "print('spectra.npz:', (DATASET_DIR / 'spectra.npz').is_file())\n"
        ),
    ],
    "colab_01_dataset.ipynb": [
        *notebook_header(
            "# Этап 1: датасет и превью\n\n"
            "**Цель:** убедиться, что датасет доступен, построить превью spectrum/structure labels.\n\n"
            "**Выход:** `runs/colab_preview/plots/preview_*.png`.\n\n"
            "Smoke: `dataset_mini` (HF). Local: версия из `paths.local.yaml`.",
            default_mode="auto",
        ),
        md_manual_dataset_upload(),
        extract_manual_datasets_cell(),
        code(
            "from pathlib import Path\n"
            "from IPython.display import Image, display\n"
            "from ir_pipeline.dataset_preview import plot_dataset_preview\n\n"
            "OUT = RUNS_DIR / 'colab_preview' / 'plots'\n"
            "plot_dataset_preview(DATASET_DIR, OUT, BANDS_YAML)\n"
            "plots = sorted(OUT.glob('preview_spectrum_*.png')) or sorted(Path('runs').rglob('preview_spectrum_0.png'))\n"
            "if plots:\n"
            "    display(Image(filename=str(plots[-1]), width=900))\n"
            "else:\n"
            "    print('Превью PNG не найдены в', OUT)\n"
        ),
    ],
    "colab_02_baseline_rf.ipynb": [
        *notebook_header(
            "# Этап 2: baseline RandomForest\n\n"
            "**Цель:** обучить RF (`spectrum_structure`) и построить графики MAE.\n\n"
            "**Выход:** `runs/colab_pipeline_rf/rf_run/` + `metrics_*.png`.\n\n"
            "Рекомендуется smoke (`dataset_mini`) или local mini/v003.",
            default_mode="auto",
        ),
        code(
            "from pathlib import Path\n"
            "from IPython.display import Image, display\n\n"
            "RUN_DIR = RUNS_DIR / 'colab_pipeline_rf' / 'rf_run'\n"
            "dv = paths['dataset_version']\n"
            "print('train:', PATHS_YAML, 'version=', dv)\n"
            "!ir-pipeline train --paths {PATHS_YAML} --dataset-version {dv} "
            "--mode spectrum_structure --config configs/train_mini.yaml --run-dir {RUN_DIR}\n"
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
        *notebook_header(
            "# Этап 3: IrResnet4 multi-label\n\n"
            "**Цель:** обучить IrResnet4 (3-канальный вход + контекст измерения).\n\n"
            "**Выход:** `runs/colab_pipeline_irresnet/irresnet_run/` (`irresnet_bundle.pt`).\n\n"
            "Предпочтительно `dataset_v003` (local/Drive); smoke — mini.",
            default_mode="auto",
            full_version="dataset_v003",
        ),
        md_cnn_hyperparameters(),
        code(
            "%matplotlib inline\n"
            "from pathlib import Path\n"
            "from ir_pipeline.config_loader import load_yaml, merge_train_defaults\n"
            "from ir_pipeline.irresnet_train import train_irresnet_run\n\n"
            "train_cfg = merge_train_defaults(load_yaml(Path('configs/train_irresnet_colab.yaml')))\n"
            "RUN_DIR = RUNS_DIR / 'colab_pipeline_irresnet' / 'irresnet_run'\n"
            "summary = train_irresnet_run(\n"
            "    dataset_dir=DATASET_DIR,\n"
            "    run_dir=RUN_DIR,\n"
            "    bands_yaml=BANDS_YAML,\n"
            "    train_cfg=train_cfg,\n"
            "    label_schema='structure_smarts',\n"
            ")\n"
            "print(summary)\n"
        ),
        md_download_run(),
        download_run_zip_cell("runs/colab_pipeline_irresnet/irresnet_run", "irresnet_run_colab"),
    ],
    "colab_04_gradcam.ipynb": [
        *notebook_header(
            "# Этап 4: Grad-CAM\n\n"
            "**Цель:** карты важности на спектре. Нужен обученный `irresnet_bundle.pt` "
            "(этап 3) или обучение mini здесь.\n\n"
            "**Выход:** `reports/colab_gradcam/gradcam_*.png`.",
            default_mode="auto",
        ),
        code(
            "from pathlib import Path\n"
            "from IPython.display import Image, display\n\n"
            "dv = paths['dataset_version']\n"
            "PIPELINE_RUN = RUNS_DIR / 'colab_pipeline_gradcam'\n"
            "bundles = sorted(Path('runs').rglob('irresnet_bundle.pt'))\n"
            "if not bundles:\n"
            "    print('Нет модели — тренируем mini/текущий датасет...')\n"
            "    !ir-pipeline irresnet-train --paths {PATHS_YAML} "
            "--dataset-version {dv} --config configs/train_irresnet.yaml "
            "--run-dir {PIPELINE_RUN / 'irresnet_run'}\n"
            "    bundles = sorted(Path('runs').rglob('irresnet_bundle.pt'))\n\n"
            "IR_RUN = bundles[-1].parent\n"
            "OUT = Path('reports/colab_gradcam')\n"
            "# Ручной режим — раскомментируйте:\n"
            "# !ir-pipeline gradcam-examples --paths {PATHS_YAML} --run-dir {IR_RUN} "
            "--output-dir {OUT} --spectrum-indices 0,7,15 --class-indices 3,12\n"
            "!ir-pipeline gradcam-examples --paths {PATHS_YAML} "
            "--run-dir {IR_RUN} --output-dir {OUT} --n-examples 3\n\n"
            "for p in sorted(OUT.glob('gradcam_*.png'))[:3]:\n"
            "    display(Image(filename=str(p), width=900))\n"
        ),
        md_download_run(),
        code(
            "# Только Colab. Локально пропустите.\n"
            "from pathlib import Path\n"
            "import shutil\n\n"
            "try:\n"
            "    from google.colab import files\n"
            "except ImportError:\n"
            "    print('Не Colab — пропуск скачивания')\n"
            "else:\n"
            "    bundles = sorted(Path('runs').rglob('irresnet_bundle.pt'))\n"
            "    if not bundles:\n"
            "        raise FileNotFoundError('Нет irresnet_bundle.pt')\n"
            "    RUN_DIR = bundles[-1].parent\n"
            "    zip_path = Path('/content/irresnet_gradcam_run_colab.zip')\n"
            "    if zip_path.exists():\n"
            "        zip_path.unlink()\n"
            "    shutil.make_archive(str(zip_path.with_suffix('')), 'zip', RUN_DIR)\n"
            "    print(f'Архив: {zip_path}')\n"
            "    files.download(str(zip_path))\n"
        ),
    ],
    "colab_05_irresnet_experiments.ipynb": [
        *notebook_header(
            "# Этап 5: сравнение IrResnet4 (hidden=72)\n\n"
            "**Цель:** E1–E4 — SMARTS-only vs SMARTS+peak × контекст on/off.\n\n"
            "**Вход:** датасет с `labels_structure_smarts.parquet` "
            "(лучше `dataset_v003`; допускается v002).\n\n"
            "**Выход:** `runs/exp_v003/summary.json`, bar chart F1.",
            default_mode="auto",
            full_version="dataset_v003",
        ),
        code("%matplotlib inline\n"),
        code(
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "from ir_pipeline.dataset_preview import build_multilabel_matrix\n"
            "from ir_pipeline.resnet_input import load_model_inputs\n\n"
            "DATASET = DATASET_DIR\n"
            "_, _, spec_ids, _, _ = load_model_inputs(DATASET)\n"
            "bands = BANDS_YAML\n"
            "rows = []\n"
            "for schema in ['structure_smarts', 'structure', 'spectrum']:\n"
            "    try:\n"
            "        Y, _ = build_multilabel_matrix(DATASET, spec_ids, bands, label_schema=schema)\n"
            "        rows.append({'schema': schema, 'mean_labels': float(Y.sum(axis=1).mean()), 'positives': int(Y.sum())})\n"
            "    except FileNotFoundError as e:\n"
            "        rows.append({'schema': schema, 'error': str(e)})\n"
            "pd.DataFrame(rows)\n"
        ),
        code(
            "from copy import deepcopy\n"
            "from pathlib import Path\n"
            "from ir_pipeline.config_loader import load_yaml, merge_train_defaults\n"
            "from ir_pipeline.irresnet_train import train_irresnet_run\n\n"
            "base_cfg = merge_train_defaults(load_yaml(Path('configs/train_irresnet_experiments.yaml')))\n"
            "EXPERIMENTS = [\n"
            "    ('E1', 'structure_smarts', False),\n"
            "    ('E2', 'structure', False),\n"
            "    ('E3', 'structure_smarts', True),\n"
            "    ('E4', 'structure', True),\n"
            "]\n"
            "EXP_ROOT = RUNS_DIR / 'exp_v003'\n"
            "summaries = []\n"
            "for exp_id, schema, use_ctx in EXPERIMENTS:\n"
            "    cfg = deepcopy(base_cfg)\n"
            "    cfg['use_measurement_context'] = use_ctx\n"
            "    run_dir = EXP_ROOT / f'{exp_id.lower()}_h72_{\"ctx\" if use_ctx else \"noctx\"}_{schema}'\n"
            "    print('===', exp_id, schema, 'context=', use_ctx, '=>', run_dir)\n"
            "    s = train_irresnet_run(\n"
            "        dataset_dir=DATASET,\n"
            "        run_dir=run_dir,\n"
            "        bands_yaml=bands,\n"
            "        train_cfg=cfg,\n"
            "        label_schema=schema,\n"
            "        use_measurement_context=use_ctx,\n"
            "    )\n"
            "    s['experiment'] = exp_id\n"
            "    summaries.append(s)\n"
            "pd.DataFrame(summaries)\n"
        ),
        code(
            "import matplotlib.pyplot as plt\n"
            "from pathlib import Path\n"
            "import pandas as pd\n\n"
            "df = pd.DataFrame(summaries)\n"
            "out = RUNS_DIR / 'exp_v003'\n"
            "out.mkdir(parents=True, exist_ok=True)\n"
            "(out / 'summary.json').write_text(df.to_json(orient='records', indent=2), encoding='utf-8')\n"
            "fig, ax = plt.subplots(figsize=(8, 4))\n"
            "x = range(len(df))\n"
            "ax.bar(x, df['test_f1_weighted'], color='steelblue')\n"
            "ax.set_xticks(list(x))\n"
            "ax.set_xticklabels(df['experiment'], rotation=0)\n"
            "ax.set_ylabel('test F1 weighted')\n"
            "ax.set_title('IrResnet4 experiments (hidden=72)')\n"
            "fig.tight_layout()\n"
            "fig.savefig(out / 'experiments_f1_weighted.png', dpi=140)\n"
            "plt.show()\n"
            "df\n"
        ),
    ],
    "colab_07_kan_compare.ipynb": [
        *notebook_header(
            "# Этап 7: KAN vs 1D CNN (M0/M1/M2)\n\n"
            "**Цель:** сравнить IrResnet4 (M0), IrKanHybrid (M1), IrKanNet (M2).\n\n"
            "**Вход:** `dataset_v003` (+ опционально `external_sdbs/sdbs_eval.npz`).\n\n"
            "**Colab full:** выполните A + A2 + C. **Local:** B + C (`paths.local.yaml`).",
            include_drive=True,
            default_mode="auto",
            full_version="dataset_v003",
            fetch_hf_if_missing=False,
        ),
        code(
            "# Пути SDBS holdout и runs на Drive (если A2 выполнен)\n"
            "from pathlib import Path\n\n"
            "if 'IR_DATA' in globals():\n"
            "    SDSBS_EVAL = Path(IR_DATA) / 'external_sdbs' / 'sdbs_eval.npz'\n"
            "    if 'RUNS_DRIVE' not in globals():\n"
            "        RUNS_DRIVE = Path('/content/drive/MyDrive/ir_expert_system_3/runs')\n"
            "else:\n"
            "    SDSBS_EVAL = Path('data/external_sdbs/sdbs_eval.npz')\n"
            "    RUNS_DRIVE = None\n"
            "print('SDSBS_EVAL', SDSBS_EVAL, SDSBS_EVAL.is_file())\n"
            "assert (DATASET_DIR / 'model_inputs.npz').is_file(), DATASET_DIR\n"
        ),
        md("## Данные и DataLoader (smoke test)"),
        code(
            "%matplotlib inline\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n"
            "import torch\n"
            "from torch.utils.data import DataLoader\n\n"
            "from ir_pipeline.dataset_preview import build_multilabel_matrix\n"
            "from ir_pipeline.dataset_split import load_split_ids\n"
            "from ir_pipeline.irresnet_train import IrDataset\n"
            "from ir_pipeline.resnet_input import load_model_inputs\n\n"
            "X_in, wn, spec_ids, X_ctx, ctx_cols = load_model_inputs(DATASET_DIR)\n"
            "Y, class_names = build_multilabel_matrix(\n"
            "    DATASET_DIR, spec_ids, BANDS_YAML, label_schema='structure_smarts'\n"
            ")\n"
            "splits = load_split_ids(DATASET_DIR)\n"
            "tr_idx = [i for i, s in enumerate(spec_ids) if s in splits['train']]\n"
            "ds = IrDataset(X_in[tr_idx[:64]], Y[tr_idx[:64]], X_ctx[tr_idx[:64]] if X_ctx is not None else None)\n"
            "dl = DataLoader(ds, batch_size=8, shuffle=True)\n"
            "xb, cb, yb = next(iter(dl))\n"
            "print('batch X', xb.shape, 'context', cb.shape, 'Y', yb.shape, 'positives', yb.sum().item())\n\n"
            "i0 = tr_idx[0]\n"
            "fig, ax = plt.subplots(1, 3, figsize=(12, 3))\n"
            "for ch, title in enumerate(['wavenumbers', 'absorption', 'peaks']):\n"
            "    ax[ch].plot(wn, X_in[i0, ch])\n"
            "    ax[ch].set_title(title)\n"
            "    ax[ch].invert_xaxis()\n"
            "fig.tight_layout()\n"
            "plt.show()\n"
        ),
        md(
            "## Модели M0/M1/M2\n\n"
            "Код из пакета (`ir_pipeline.models`). Правки: `src/ir_pipeline/models/` + "
            "`pip install -e .[torch]`."
        ),
        code(
            "from ir_pipeline.models.model_factory import build_spectrum_model, MODEL_FAMILIES\n\n"
            "print('families:', MODEL_FAMILIES)\n"
        ),
        md("## Обучение одной модели и сравнение всех трёх"),
        code(
            "import json\n"
            "import shutil\n"
            "from copy import deepcopy\n"
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "from ir_pipeline.config_loader import load_yaml, merge_train_defaults\n"
            "from ir_pipeline.irresnet_train import train_irresnet_run\n\n"
            "TRAIN_CFG = merge_train_defaults(load_yaml(Path('configs/train_kan_colab.yaml')))\n"
            "HIDDEN_SIZE = int(TRAIN_CFG.get('compare_hidden_size', TRAIN_CFG.get('ir_hidden_size', 34)))\n"
            "RUN_TAG = f'kan_cmp_h{HIDDEN_SIZE}'\n\n"
            "FAMILY_LABELS = {\n"
            "    'irresnet4': 'M0',\n"
            "    'kan_hybrid': 'M1',\n"
            "    'kan_full': 'M2',\n"
            "}\n\n"
            "def train_all_models(\n"
            "    hidden_size: int,\n"
            "    *,\n"
            "    dataset_dir: Path,\n"
            "    run_root: Path,\n"
            "    train_cfg: dict,\n"
            "    model_families: tuple[str, ...] = ('irresnet4', 'kan_hybrid', 'kan_full'),\n"
            "    save_to_drive: Path | None = None,\n"
            "    download_zip: bool = False,\n"
            ") -> pd.DataFrame:\n"
            "    run_root.mkdir(parents=True, exist_ok=True)\n"
            "    summaries, histories, run_dirs = [], {}, {}\n"
            "    for fam in model_families:\n"
            "        label = FAMILY_LABELS.get(fam, fam)\n"
            "        sub = run_root / f'{label.lower()}_{fam}'\n"
            "        cfg = deepcopy(train_cfg)\n"
            "        cfg['model_family'] = fam\n"
            "        cfg['ir_hidden_size'] = hidden_size\n"
            "        cfg['kan_full_hidden_size'] = hidden_size\n"
            "        print('===', label, fam, 'hidden=', hidden_size, '=>', sub)\n"
            "        summary = train_irresnet_run(\n"
            "            dataset_dir=dataset_dir,\n"
            "            run_dir=sub,\n"
            "            bands_yaml=BANDS_YAML,\n"
            "            train_cfg=cfg,\n"
            "        )\n"
            "        summary['model_label'] = label\n"
            "        summaries.append(summary)\n"
            "        run_dirs[label] = sub\n"
            "        hist_path = sub / 'irresnet_history.json'\n"
            "        if hist_path.is_file():\n"
            "            histories[label] = json.loads(hist_path.read_text(encoding='utf-8'))\n"
            "    df = pd.DataFrame(summaries)\n"
            "    (run_root / 'compare_summary.json').write_text(\n"
            "        df.to_json(orient='records', indent=2), encoding='utf-8'\n"
            "    )\n"
            "    if save_to_drive:\n"
            "        dest = save_to_drive / run_root.name\n"
            "        if dest.exists():\n"
            "            shutil.rmtree(dest)\n"
            "        shutil.copytree(run_root, dest)\n"
            "        print('saved to Drive:', dest)\n"
            "    if download_zip:\n"
            "        try:\n"
            "            from google.colab import files\n"
            "            zip_base = Path('/content') / run_root.name\n"
            "            if Path(str(zip_base) + '.zip').exists():\n"
            "                Path(str(zip_base) + '.zip').unlink()\n"
            "            archive = shutil.make_archive(str(zip_base), 'zip', run_root)\n"
            "            files.download(archive)\n"
            "        except ImportError:\n"
            "            print('download_zip: не Colab')\n"
            "    globals()['COMPARE_HISTORIES'] = histories\n"
            "    globals()['COMPARE_RUN_DIRS'] = run_dirs\n"
            "    return df\n"
        ),
        code(
            "results_df = train_all_models(\n"
            "    hidden_size=HIDDEN_SIZE,\n"
            "    dataset_dir=DATASET_DIR,\n"
            "    run_root=RUNS_DIR / RUN_TAG,\n"
            "    train_cfg=TRAIN_CFG,\n"
            "    save_to_drive=(RUNS_DRIVE / RUN_TAG) if RUNS_DRIVE else None,\n"
            "    download_zip=False,\n"
            ")\n"
            "display(results_df[['model_label', 'model_family', 'n_params', 'test_f1_weighted', 'test_lrap', 'train_wall_time_sec']])\n"
        ),
        md("## Графики обучения (overlay M0/M1/M2)"),
        code(
            "def plot_training_comparison(histories: dict, out_path: Path | None = None):\n"
            "    fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
            "    for label, hist in histories.items():\n"
            "        epochs = range(1, len(hist.get('train_loss', [])) + 1)\n"
            "        axes[0].plot(epochs, hist['train_loss'], label=f'{label} train')\n"
            "        axes[0].plot(epochs, hist['val_loss'], ls='--', label=f'{label} val')\n"
            "        if 'val_lrap' in hist:\n"
            "            axes[1].plot(epochs, hist['val_lrap'], label=label)\n"
            "    axes[0].set_ylabel('loss')\n"
            "    axes[0].legend(fontsize=7)\n"
            "    axes[1].set_ylabel('val LRAP')\n"
            "    axes[1].legend(fontsize=7)\n"
            "    fig.suptitle('KAN vs CNN training curves')\n"
            "    fig.tight_layout()\n"
            "    if out_path:\n"
            "        fig.savefig(out_path, dpi=140)\n"
            "    plt.show()\n\n"
            "plot_training_comparison(\n"
            "    COMPARE_HISTORIES,\n"
            "    out_path=RUNS_DIR / RUN_TAG / 'training_curves.png',\n"
            ")\n"
        ),
        md("## Внешний тест: SDBS (вне dataset_v003)"),
        code(
            "def load_sdbs_eval(npz_path: Path):\n"
            "    z = np.load(npz_path, allow_pickle=True)\n"
            "    v003_ids = set(spec_ids)\n"
            "    ext_ids = [str(s) for s in z['spectrum_id']]\n"
            "    overlap = [s for s in ext_ids if s in v003_ids]\n"
            "    if overlap:\n"
            "        print('WARNING: overlap with v003:', len(overlap))\n"
            "    return z['X_input'], ext_ids, z\n\n"
            "if SDSBS_EVAL.is_file():\n"
            "    X_ext, ext_ids, sdbs_z = load_sdbs_eval(SDSBS_EVAL)\n"
            "    print('external spectra:', X_ext.shape)\n"
            "else:\n"
            "    print('Нет', SDSBS_EVAL, '— python tools/build_sdbs_holdout.py')\n"
            "    X_ext, ext_ids = None, []\n"
        ),
        md("## Инференс и визуализация внимания"),
        code(
            "import torch.nn.functional as F\n"
            "from ir_pipeline.gradcam import compute_cam, interpolate_cam_to_wavenumbers\n"
            "from ir_pipeline.models.model_factory import build_spectrum_model\n\n"
            "def load_bundle_model(bundle_path: Path):\n"
            "    ck = torch.load(bundle_path, map_location='cpu', weights_only=False)\n"
            "    meta = ck['meta']\n"
            "    model = build_spectrum_model(\n"
            "        meta.get('model_family', 'irresnet4'),\n"
            "        hidden_size=int(meta['hidden_size']),\n"
            "        class_nums=len(meta['class_names']),\n"
            "        context_dim=int(meta.get('context_dim', 0)),\n"
            "        train_cfg={\n"
            "            'kan_grid_size': meta.get('kan_grid_size'),\n"
            "            'kan_full_hidden_size': meta.get('kan_full_hidden_size', meta['hidden_size']),\n"
            "        },\n"
            "    )\n"
            "    model.load_state_dict(ck['model_state'])\n"
            "    model.eval()\n"
            "    return model, meta\n\n"
            "def run_external_inference(bundles, X_ext, class_names, threshold=0.5):\n"
            "    rows = []\n"
            "    for label, bpath in bundles.items():\n"
            "        model, meta = load_bundle_model(bpath)\n"
            "        with torch.no_grad():\n"
            "            logits = model(torch.from_numpy(X_ext).float())\n"
            "            prob = torch.sigmoid(logits).numpy()\n"
            "        for i in range(len(X_ext)):\n"
            "            top = np.argsort(-prob[i])[:5]\n"
            "            rows.append({\n"
            "                'model': label,\n"
            "                'spectrum_idx': i,\n"
            "                'top_bands': [class_names[j] for j in top],\n"
            "                'top_probs': [float(prob[i, j]) for j in top],\n"
            "            })\n"
            "    return pd.DataFrame(rows)\n\n"
            "def plot_model_attention(model, x_3ch, class_idx, wavenumbers, band_range=None):\n"
            "    x = torch.from_numpy(x_3ch).float().unsqueeze(0)\n"
            "    cam = compute_cam(model, x, class_idx)\n"
            "    cam_i = interpolate_cam_to_wavenumbers(cam, wavenumbers)\n"
            "    cam_i = cam_i / (cam_i.max() + 1e-9)\n"
            "    ab = x_3ch[1]\n"
            "    fig, ax = plt.subplots(figsize=(11, 3))\n"
            "    ax.plot(wavenumbers, ab, color='gray', lw=1)\n"
            "    ax.fill_between(wavenumbers, 0, cam_i * ab.max(), alpha=0.35, color='crimson')\n"
            "    if band_range:\n"
            "        ax.axvspan(band_range[0], band_range[1], color='green', alpha=0.1)\n"
            "    ax.invert_xaxis()\n"
            "    ax.set_xlabel(r'cm$^{-1}$')\n"
            "    ax.set_title(f'Grad-CAM class_idx={class_idx}')\n"
            "    fig.tight_layout()\n"
            "    plt.show()\n"
            "    return cam_i\n"
        ),
        code(
            "if X_ext is not None and 'COMPARE_RUN_DIRS' in globals():\n"
            "    bundles = {k: v / 'irresnet_bundle.pt' for k, v in COMPARE_RUN_DIRS.items()}\n"
            "    bundles = {k: v for k, v in bundles.items() if v.is_file()}\n"
            "    if bundles:\n"
            "        inf_df = run_external_inference(bundles, X_ext[:5], class_names)\n"
            "        display(inf_df.head(15))\n"
            "        spec_i = 0\n"
            "        for label, bpath in bundles.items():\n"
            "            model, meta = load_bundle_model(bpath)\n"
            "            with torch.no_grad():\n"
            "                logits = model(torch.from_numpy(X_ext[spec_i:spec_i+1]).float())\n"
            "                cidx = int(torch.sigmoid(logits)[0].argmax())\n"
            "            print('---', label, 'top class:', class_names[cidx])\n"
            "            plot_model_attention(model, X_ext[spec_i], cidx, wn)\n"
            "else:\n"
            "    print('Пропуск инференса: нет SDBS или не обучены модели')\n"
        ),
    ],
}


if __name__ == "__main__":
    for name, cells in NOTEBOOKS.items():
        path = ROOT / name
        path.write_text(json.dumps(nb(cells), ensure_ascii=False, indent=1), encoding="utf-8")
        print("wrote", path)
