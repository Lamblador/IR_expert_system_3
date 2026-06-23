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
        "| **Метки** | `structure` / `structure_smarts` / `spectrum` | `--label-schema` в CLI или kwarg в `train_irresnet_run` |\n"
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
        "import sys\n"
        "from pathlib import Path\n\n"
        "REPO_URL = \"https://github.com/Lamblador/IR_expert_system_3.git\"\n"
        "REPO_BRANCH = \"colab-v1\"\n"
        "REPO_DIR_NAME = \"IR_expert_system_3\"\n\n"
        "def _run_git(cmd: list[str], cwd: Path | None = None) -> None:\n"
        "    print('git', ' '.join(cmd), f'(cwd={cwd})' if cwd else '')\n"
        "    subprocess.run(cmd, cwd=cwd, check=True)\n\n"
        "def _ensure_repo_at(repo_dir: Path) -> None:\n"
        "    if (repo_dir / '.git').is_dir():\n"
        "        _run_git(['git', 'fetch', 'origin', REPO_BRANCH], cwd=repo_dir)\n"
        "        _run_git(['git', 'checkout', REPO_BRANCH], cwd=repo_dir)\n"
        "        _run_git(['git', 'pull', '--ff-only', 'origin', REPO_BRANCH], cwd=repo_dir)\n"
        "    else:\n"
        "        if repo_dir.exists():\n"
        "            raise RuntimeError(f'{repo_dir} существует, но это не git-репозиторий')\n"
        "        _run_git([\n"
        "            'git', 'clone', '-b', REPO_BRANCH, '--single-branch',\n"
        "            REPO_URL, str(repo_dir),\n"
        "        ])\n"
        "    rev = subprocess.check_output(\n"
        "        ['git', 'rev-parse', '--short', 'HEAD'], cwd=repo_dir, text=True\n"
        "    ).strip()\n"
        "    print(f'Репозиторий: {repo_dir.resolve()} @ {REPO_BRANCH} ({rev})')\n\n"
        "cwd = Path.cwd()\n"
        "in_colab = Path('/content').exists() and str(cwd).startswith('/content')\n"
        "local_repo = (cwd / 'pyproject.toml').is_file()\n\n"
        "if local_repo and not in_colab:\n"
        "    ROOT = cwd.resolve()\n"
        "    print('Локальный репозиторий (ветку не переключаем):', ROOT)\n"
        "elif (cwd / REPO_DIR_NAME / 'pyproject.toml').is_file():\n"
        "    ROOT = (cwd / REPO_DIR_NAME).resolve()\n"
        "    _ensure_repo_at(ROOT)\n"
        "elif Path(f'/content/{REPO_DIR_NAME}/pyproject.toml').is_file():\n"
        "    ROOT = Path(f'/content/{REPO_DIR_NAME}').resolve()\n"
        "    _ensure_repo_at(ROOT)\n"
        "else:\n"
        "    ROOT = (Path('/content') / REPO_DIR_NAME if in_colab else cwd / REPO_DIR_NAME).resolve()\n"
        "    _ensure_repo_at(ROOT)\n\n"
        "import os\n"
        "os.chdir(ROOT)\n"
        "try:\n"
        "    from IPython import get_ipython\n"
        "    get_ipython().run_line_magic('cd', str(ROOT))\n"
        "except Exception:\n"
        "    pass\n"
        "print('ROOT:', ROOT.resolve())\n"
        "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', '.[torch]'], check=True)\n"
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
        md("# Этап 3: IrResnet4 multi-label (автономный)\n\n3-канальный вход 400–4000 см⁻¹ + контекст ATR/gas/solution."),
        bootstrap_cell(),
        mount_google_drive_cell(),
        md_manual_dataset_upload(),
        extract_manual_datasets_cell(),
        ensure_data_cell(),
        md_cnn_hyperparameters(),
        code(
            "%matplotlib inline\n"
            "from pathlib import Path\n"
            "from ir_pipeline.config_loader import load_yaml, merge_train_defaults, resolve_paths\n"
            "from ir_pipeline.irresnet_train import train_irresnet_run\n\n"
            "paths = resolve_paths(load_yaml(Path('configs/paths.huggingface.yaml')))\n"
            "train_cfg = merge_train_defaults(load_yaml(Path('configs/train_irresnet_colab.yaml')))\n"
            "DATASET = paths['processed_root'] / 'dataset_v002'  # или dataset_mini\n"
            "RUN_DIR = Path('runs/colab_pipeline_irresnet/irresnet_run')\n"
            "summary = train_irresnet_run(\n"
            "    dataset_dir=DATASET,\n"
            "    run_dir=RUN_DIR,\n"
            "    bands_yaml=paths['bands_config'],\n"
            "    train_cfg=train_cfg,\n"
            "    label_schema='structure_smarts',\n"
            ")\n"
            "print(summary)\n"
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
    "colab_05_irresnet_experiments.ipynb": [
        md(
            "# Этап 5: сравнение IrResnet4 (hidden=72)\n\n"
            "E1–E4: SMARTS-only vs SMARTS+peak × контекст измерения on/off.\n"
            "Требуется `dataset_v002` с `labels_structure_smarts.parquet`."
        ),
        bootstrap_cell(),
        mount_google_drive_cell(),
        md_manual_dataset_upload(),
        extract_manual_datasets_cell(),
        ensure_data_cell(),
        code("%matplotlib inline\n"),
        code(
            "from pathlib import Path\n"
            "import json\n"
            "import pandas as pd\n"
            "from ir_pipeline.config_loader import load_yaml, merge_train_defaults, resolve_paths\n"
            "from ir_pipeline.dataset_preview import build_multilabel_matrix\n"
            "from ir_pipeline.resnet_input import load_model_inputs\n\n"
            "paths = resolve_paths(load_yaml(Path('configs/paths.huggingface.yaml')))\n"
            "DATASET = paths['processed_root'] / 'dataset_v002'\n"
            "_, _, spec_ids, _, _ = load_model_inputs(DATASET)\n"
            "bands = paths['bands_config']\n"
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
            "summaries = []\n"
            "for exp_id, schema, use_ctx in EXPERIMENTS:\n"
            "    cfg = deepcopy(base_cfg)\n"
            "    cfg['use_measurement_context'] = use_ctx\n"
            "    run_dir = Path('runs/exp_v002') / f'{exp_id.lower()}_h72_{\"ctx\" if use_ctx else \"noctx\"}_{schema}'\n"
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
            "out = Path('runs/exp_v002')\n"
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
        md(
            "# KAN vs 1D CNN: обучение M0/M1/M2 в Colab\n\n"
            "Сравнение **IrResnet4** (M0), **IrKanHybrid** (M1), **IrKanNet** (M2) при одинаковом `hidden_size`.\n\n"
            "**Drive:** `ir_data/processed/dataset_v003`, `ir_data/external_sdbs/sdbs_eval.npz`, "
            "`ir_expert_system_3/runs/`."
        ),
        bootstrap_cell(),
        code(
            "from pathlib import Path\n"
            "from google.colab import drive\n\n"
            "drive.mount('/content/drive')\n"
            "IR_DATA = Path('/content/drive/MyDrive/ir_data')\n"
            "RUNS_DRIVE = Path('/content/drive/MyDrive/ir_expert_system_3/runs')\n"
            "SDSBS_EVAL = IR_DATA / 'external_sdbs' / 'sdbs_eval.npz'\n"
            "RUNS_DRIVE.mkdir(parents=True, exist_ok=True)\n"
            "print('IR_DATA', IR_DATA.exists())\n"
            "print('SDSBS_EVAL', SDSBS_EVAL.exists())\n"
        ),
        code(
            "import os\n"
            "from pathlib import Path\n"
            "from ir_pipeline.config_loader import load_yaml, merge_train_defaults, resolve_paths\n\n"
            "os.environ['IR_PROCESSED_ROOT'] = str(IR_DATA / 'processed')\n"
            "paths_cfg = load_yaml(Path('configs/paths.colab.yaml'))\n"
            "paths_cfg['dataset_version'] = 'dataset_v003'\n"
            "paths = resolve_paths(paths_cfg)\n"
            "DATASET_DIR = paths['processed_root'] / paths['dataset_version']\n"
            "BANDS_YAML = paths['bands_config']\n"
            "assert (DATASET_DIR / 'model_inputs.npz').is_file(), DATASET_DIR\n"
            "print('dataset:', DATASET_DIR)\n"
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
            "Код импортируется из пакета (`ir_pipeline.models`). Для экспериментов редактируйте "
            "`src/ir_pipeline/models/` и выполните `pip install -e .[torch]`."
        ),
        code(
            "from ir_pipeline.models.kan_layers import KANLinear, ConvKAN1d, KANHead, KanBasicBlock\n"
            "from ir_pipeline.models.ir_resnet4 import IrResnet4\n"
            "from ir_pipeline.models.ir_kan_hybrid import IrKanHybrid\n"
            "from ir_pipeline.models.ir_kan_net import IrKanNet\n"
            "from ir_pipeline.models.model_factory import build_spectrum_model, count_parameters, MODEL_FAMILIES\n\n"
            "def build_model(family, hidden_size, n_classes, context_dim=12):\n"
            "    return build_spectrum_model(\n"
            "        family,\n"
            "        hidden_size=hidden_size,\n"
            "        class_nums=n_classes,\n"
            "        context_dim=context_dim,\n"
            "        train_cfg=TRAIN_CFG,\n"
            "    )\n\n"
            "print('families:', MODEL_FAMILIES)\n"
        ),
        md("## Обучение одной модели и сравнение всех трёх"),
        code(
            "import json\n"
            "import shutil\n"
            "import time\n"
            "from copy import deepcopy\n"
            "from pathlib import Path\n"
            "import pandas as pd\n"
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
            "    summaries = []\n"
            "    histories = {}\n"
            "    run_dirs = {}\n"
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
            "        from google.colab import files\n"
            "        zip_base = Path('/content') / run_root.name\n"
            "        if Path(str(zip_base) + '.zip').exists():\n"
            "            Path(str(zip_base) + '.zip').unlink()\n"
            "        archive = shutil.make_archive(str(zip_base), 'zip', run_root)\n"
            "        files.download(archive)\n"
            "    globals()['COMPARE_HISTORIES'] = histories\n"
            "    globals()['COMPARE_RUN_DIRS'] = run_dirs\n"
            "    return df\n"
        ),
        code(
            "results_df = train_all_models(\n"
            "    hidden_size=HIDDEN_SIZE,\n"
            "    dataset_dir=DATASET_DIR,\n"
            "    run_root=Path('runs') / RUN_TAG,\n"
            "    train_cfg=TRAIN_CFG,\n"
            "    save_to_drive=RUNS_DRIVE / RUN_TAG,\n"
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
            "    out_path=Path('runs') / RUN_TAG / 'training_curves.png',\n"
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
            "    print('Нет', SDSBS_EVAL, '— соберите локально: python tools/build_sdbs_holdout.py')\n"
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
            "def run_external_inference(bundles: dict[str, Path], X_ext: np.ndarray, class_names: list[str], threshold=0.5):\n"
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
