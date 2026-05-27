"""Генератор Colab-ноутбуков (запускать один раз при изменении шаблонов)."""
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
    return {"cell_type": "code", "metadata": {}, "source": source.splitlines(keepends=True), "outputs": [], "execution_count": None}


NOTEBOOKS = {
    "colab_00_setup.ipynb": [
        md(
            "# Этап 0: установка\n\n"
            "Клонируем репозиторий, ставим пакет и опционально PyTorch.\n\n"
            "**Следующий ноутбук:** `colab_01_dataset.ipynb`"
        ),
        code(
            "import subprocess\nfrom pathlib import Path\n\n"
            "REPO_URL = \"https://github.com/Lamblador/IR_expert_system_3.git\"  # замените на свой форк\n"
            "REPO_DIR = Path(\"IR_expert_system_3\")\n\n"
            "if not REPO_DIR.is_dir():\n"
            "    subprocess.run([\"git\", \"clone\", REPO_URL, str(REPO_DIR)], check=True)\n"
            "%cd IR_expert_system_3\n"
            "!pip install -q -e \".[torch]\"\n"
            "!ir-pipeline run list\n"
        ),
    ],
    "colab_01_dataset.ipynb": [
        md(
            "# Этап 1: датасет\n\n"
            "Скачивание mini-dataset с HF и превью (спектры, баланс классов).\n\n"
            "**Ожидаемые артефакты:** `data/processed/dataset_mini/`, PNG в `runs/.../stage_03_.../plots/`"
        ),
        code(
            "%cd IR_expert_system_3\n"
            "import os\nfrom pathlib import Path\nfrom IPython.display import Image, display\n\n"
            "PIPELINE_RUN = Path(\"runs/colab_pipeline_dataset\")\n"
            "!ir-pipeline run stage fetch --paths configs/paths.huggingface.yaml --pipeline-run {PIPELINE_RUN}\n"
            "!ir-pipeline run stage dataset_preview --paths configs/paths.huggingface.yaml --pipeline-run {PIPELINE_RUN}\n"
        ),
        code(
            "from pathlib import Path\nplots = sorted(Path(\"runs\").rglob(\"preview_spectrum_0.png\"))\n"
            "if plots:\n    display(Image(filename=str(plots[-1]), width=900))\nelse:\n    print(\"Нет preview PNG — проверьте error_log.txt в stage_03\")\n"
        ),
    ],
    "colab_02_baseline_rf.ipynb": [
        md(
            "# Этап 2: baseline RandomForest\n\n"
            "Обучение RF (CPU sklearn или GPU cuML) + графики MAE.\n\n"
            "**Следующий:** `colab_03_train_irresnet4.ipynb`"
        ),
        code(
            "%cd IR_expert_system_3\n"
            "from pathlib import Path\nPIPELINE_RUN = Path(\"runs/colab_pipeline_rf\")\n"
            "# GPU (раскомментируйте при Runtime → GPU):\n"
            "# !pip install -q -e \".[cuml]\"\n"
            "# import os; os.environ[\"IR_RF_BACKEND\"] = \"cuml\"\n"
            "!ir-pipeline run stage train_rf --paths configs/paths.huggingface.yaml --pipeline-run {PIPELINE_RUN}\n"
            "!ir-pipeline run stage plot_rf_metrics --paths configs/paths.huggingface.yaml --pipeline-run {PIPELINE_RUN}\n"
        ),
        code(
            "from pathlib import Path\nfrom IPython.display import Image, display\n"
            "for pat in [\"metrics_per_band_mae.png\", \"metrics_by_group_mae.png\"]:\n"
            "    hits = sorted(Path(\"runs\").rglob(pat))\n"
            "    if hits:\n        print(hits[-1]); display(Image(filename=str(hits[-1]), width=900))\n"
        ),
    ],
    "colab_03_train_irresnet4.ipynb": [
        md(
            "# Этап 3: IrResnet4 (multi-label, CAM-ready)\n\n"
            "Формат входа совместим с FTIR Telegram-ботом (3 канала, 500–4100 см⁻¹)."
        ),
        code(
            "%cd IR_expert_system_3\n"
            "from pathlib import Path\nPIPELINE_RUN = Path(\"runs/colab_pipeline_irresnet\")\n"
            "!ir-pipeline run stage train_irresnet --paths configs/paths.huggingface.yaml --pipeline-run {PIPELINE_RUN}\n"
        ),
        code(
            "from pathlib import Path\nfrom IPython.display import Image, display\n"
            "hits = sorted(Path(\"runs\").rglob(\"irresnet_training_curve.png\"))\n"
            "if hits:\n    display(Image(filename=str(hits[-1]), width=900))\n"
        ),
    ],
    "colab_04_cam_examples.ipynb": [
        md("# Этап 4: Grad-CAM\n\nВизуализация областей спектра, влияющих на предсказание классов."),
        code(
            "%cd IR_expert_system_3\n"
            "from pathlib import Path\nPIPELINE_RUN = Path(\"runs/colab_pipeline_cam\")\n"
            "# Укажите run с irresnet_bundle.pt из этапа 3:\n"
            "IR_RUN = sorted(Path(\"runs\").rglob(\"irresnet_bundle.pt\"))[-1].parent\n"
            "print(\"Using\", IR_RUN)\n"
            "!ir-pipeline cam-examples --paths configs/paths.huggingface.yaml --run-dir {IR_RUN} --output-dir reports/colab_cam\n"
        ),
        code(
            "from pathlib import Path\nfrom IPython.display import Image, display\n"
            "for p in sorted(Path(\"reports/colab_cam\").glob(\"cam_example_*.png\"))[:3]:\n"
            "    display(Image(filename=str(p), width=900))\n"
        ),
    ],
    "colab_05_export_telegram.ipynb": [
        md(
            "# Этап 5: экспорт в Telegram-бот\n\n"
            "Копирует веса в `FTIR_telegram_bot/models/` (путь на Drive или локально — поправьте `TARGET`)."
        ),
        code(
            "%cd IR_expert_system_3\n"
            "from pathlib import Path\n"
            "IR_RUN = sorted(Path(\"runs\").rglob(\"irresnet_bundle.pt\"))[-1].parent\n"
            "TARGET = Path(\"/content/drive/MyDrive/ftir_bot_models\")  # или локальный путь к боту\n"
            "TARGET.mkdir(parents=True, exist_ok=True)\n"
            "!ir-pipeline export-telegram --run-dir {IR_RUN} --target-dir {TARGET}\n"
            "print(\"Files:\", list(TARGET.rglob(\"*_model_param\")))\n"
        ),
    ],
}

if __name__ == "__main__":
    for name, cells in NOTEBOOKS.items():
        path = ROOT / name
        path.write_text(json.dumps(nb(cells), ensure_ascii=False, indent=1), encoding="utf-8")
        print("wrote", path)
