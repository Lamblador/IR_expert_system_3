# IR Pipeline — FT-IR JCAMP → ML → визуализация

Пайплайн по плану: чтение JCAMP через пакет `jcamp`, QC, приведение к оси 400–4000 см⁻¹, справочник полос с SMARTS, резолвинг структуры (CAS → SMILES через PubChem), разметка по подструктурам, кэш датасета, обучение baseline (sklearn / опционально PyTorch), метрики и вывод графика с текстовым IR-описанием.

## Установка

```bash
cd IR_expert_system_3
pip install -e .
# опционально GPU-модель:
pip install -e ".[torch]"
```

## Данные

Положите каталог `downloaded_jcamp` в корень проекта или задайте путь в `configs/paths.local.yaml` / переменной окружения `IR_RAW_JCAMP_DIR`.

Большие артефакты (`data/processed/`, модели) не коммитьте; для Colab см. `configs/paths.colab.yaml`.

Публичный Dataset repo по умолчанию: [Lamblador/IRSpectra2](https://huggingface.co/datasets/Lamblador/IRSpectra2). Скачивание идёт через библиотеку `huggingface_hub` (в том числе LFS/GCS), без ручных прямых ссылок.

```bash
# По умолчанию repo-id=Lamblador/IRSpectra2 (--repo-id можно не указывать).

# Мини-датасет для быстрой проверки Colab (~сотни спектров → архив должен содержать каталог dataset_mini/)
ir-pipeline fetch-data --filename dataset_mini.zip --extract-to data/processed

# Полный собранный датасет
ir-pipeline fetch-data --filename dataset_v001.zip --extract-to data/processed

# Сырые JCAMP → после распаковки должен появиться ./downloaded_jcamp/
ir-pipeline fetch-data --filename downloaded_jcamp.zip --extract-to .
```

Google Colab (минимум):

```python
!pip install -e .
# без токена, если репозиторий публичный:
!ir-pipeline fetch-data --filename dataset_mini.zip --extract-to data/processed
!ir-pipeline train --paths configs/paths.huggingface.yaml --config configs/train_mini.yaml --run-dir runs/mini01
```

Чтобы **`ir-pipeline train` использовал GPU** (RandomForest на CUDA через [RAPIDS cuML](https://docs.rapids.ai/api/cuml/stable/)), включите **Runtime → GPU**, затем установите cuML и при необходимости зафиксируйте бэкенд:

```python
!pip install -e ".[cuml]"
# или: !pip install cuml-cu12
import os
os.environ["IR_RF_BACKEND"] = "cuml"  # опционально; при auto выберется cuML, если есть GPU и пакет
```

При отсутствии cuML или GPU пайплайн сам перейдёт на sklearn на CPU.

Для private repo задайте токен в окружении:

```bash
export HF_TOKEN=hf_...   # Settings → Access Tokens → Read
# Windows PowerShell:
$env:HF_TOKEN = "hf_..."
```

В Colab:

```python
import os
from google.colab import userdata  # или os.environ вручную
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
```

Что выложить в файлы репозитория на Hugging Face (вкладка Files у Dataset):

- `dataset_mini.zip` — для Colab/smoke: локально `ir-pipeline build-mini-dataset --paths configs/paths.local.yaml` (это то же самое, что `build-dataset --max-files 300 --dataset-version dataset_mini`). Из `data/processed/` упакуйте **папку** `dataset_mini` в zip так, чтобы после распаковки в `data/processed` получилось `data/processed/dataset_mini/spectra.npz` и остальные parquet/json.
- `dataset_v001.zip` — полный собранный датасет; внутри каталог `dataset_v001/` с теми же артефактами.
- `downloaded_jcamp.zip` — только если нужна полная пересборка с нуля; внутри каталог `downloaded_jcamp/` с JCAMP (`*.jdx` или имена CAS без расширения).
- Опционально отдельным файлом в корне repo или в zip: `lamblador_irspectra_structures.parquet` в `data/processed/` — иначе seed подтянется сам при первом `build-dataset`.

Почему обучение sklearn на Colab кажется «очень медленным»: для режима `spectrum` обучается **отдельный RandomForest на каждую полосу** из меток (десятки моделей), каждая видит вектор признаков длины ~1801 (точки спектра) плюс категориальные признаки, деревьев по умолчанию много (`train_smoke.yaml`: 80+). Для быстрых прогонов используйте **`configs/train_mini.yaml`** и **`dataset_mini.zip`**.

**GPU и `ir-pipeline train`:** по умолчанию используется **scikit-learn RandomForest** (CPU). После `pip install -e ".[cuml]"` при доступной NVIDIA GPU команда `train` может автоматически перейти на **cuML RandomForest** (см. блок Colab выше; лог: `rf_backend=cuml`). Альтернатива — **`ir-pipeline torch-train`** после `pip install -e ".[torch]"` (другая архитектура модели). Проверка CUDA: `import cupy as cp; print(cp.cuda.runtime.getDeviceCount())` или `import torch; print(torch.cuda.is_available())`.

После `build-dataset` в `data/processed/<dataset_version>/` появляются:

- `spectra.npz`: `X` (нормализованный спектр), `X_absorbance_corrected`, `X_absorbance_like_interp`, `coverage`, `wavenumbers`
- `meta.parquet`, `labels_spectrum.parquet`, `labels_structure.parquet`, `unresolved_structures.parquet`
- `structure_cache.parquet`, `split.json` (фиксированное разбиение для обучения/валидации), `manifest.json`

## Этапы пайплайна (оркестратор)

См. [`docs/PIPELINE.md`](docs/PIPELINE.md) и [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

```bash
# Список стадий и профилей
ir-pipeline run list

# Colab / smoke: HF mini-dataset → превью → RF
ir-pipeline run profile smoke --paths configs/paths.huggingface.yaml

# Одна стадия
ir-pipeline run stage dataset_preview --paths configs/paths.local.yaml
```

Colab: ноутбуки в [`notebooks/`](notebooks/) (`colab_00_setup` … `colab_04_gradcam`).

## Команды

```bash
# Быстрая сборка версионированного датасета: только локальный/HF seed-кэш, без PubChem
ir-pipeline build-dataset --paths configs/paths.local.yaml --max-files 0

# Опционально: медленно добрать структуры, которых нет в быстром кэше, затем пересобрать датасет
ir-pipeline resolve-missing-structures --paths configs/paths.local.yaml --dataset-version dataset_v001
ir-pipeline build-dataset --paths configs/paths.local.yaml --max-files 0

# Быстрая проверка на подвыборке
ir-pipeline build-dataset --paths configs/paths.local.yaml --max-files 100 --dataset-version dataset_smoke

# Мини-датасет для HF / Colab (по умолчанию 300 файлов → dataset_mini/)
ir-pipeline build-mini-dataset --paths configs/paths.local.yaml

# Скачать с HF (публичный repo по умолчанию)
ir-pipeline fetch-data --filename dataset_mini.zip --extract-to data/processed

# Графики MAE из runs/.../metrics.json (по полосам и по функциональной группе из bands_reference)
ir-pipeline plot-train-metrics --run-dir runs/colab_cpu_rf

# Обучение RandomForest (режим только спектр или спектр + SMARTS-маска)
ir-pipeline train --paths configs/paths.huggingface.yaml --config configs/train_mini.yaml --run-dir runs/mini01
ir-pipeline train --paths configs/paths.local.yaml --dataset-version dataset_v001 --config configs/train_smoke.yaml
# только пики в регионе без SMARTS:
ir-pipeline train --paths configs/paths.local.yaml --dataset-version dataset_v001 --mode spectrum --config configs/train_smoke.yaml

# Оценка
ir-pipeline evaluate --paths configs/paths.local.yaml --dataset-version dataset_v001 --run-dir runs/run_001

# Предсказание + PNG + текстовое описание
ir-pipeline predict --paths configs/paths.local.yaml --jcamp path/to/file --run-dir runs/run_001 --output-dir reports/out

# PyTorch: многозадачная 1D CNN по позициям пиков (после pip install -e ".[torch]")
ir-pipeline torch-train --paths configs/paths.local.yaml --dataset-version dataset_smoke --config configs/train_smoke.yaml
# → torch_bundle.pt, torch_training_curve.png, torch_history.json

ir-pipeline predict ... --run-dir runs/<torch_run> --torch --output-dir reports/out_torch

# IrResnet4 multi-label + Grad-CAM (ручной выбор индексов опционален)
ir-pipeline irresnet-train --paths configs/paths.local.yaml --dataset-version dataset_mini --config configs/train_irresnet.yaml
ir-pipeline gradcam-examples --paths configs/paths.local.yaml --run-dir runs/<irresnet_run> --output-dir reports/gradcam
ir-pipeline gradcam-examples --run-dir runs/<irresnet_run> --spectrum-indices 0,7,15 --class-indices 3,12

```

Справочник [`configs/bands_reference.yaml`](configs/bands_reference.yaml) дополнен строками по таблице Википедии «[Таблица характеристических частот в инфракрасной спектроскопии](https://ru.wikipedia.org/wiki/Таблица_характеристических_частот_в_инфракрасной_спектроскопии)» (`source: ru.wikipedia IR table`); скорректированы интервалы для `ch_sp3`, замечания для `nh_stretch`.

## Конфигурация

- `configs/paths.local.yaml` — локальные пути
- `configs/paths.colab.yaml` — пример для Google Drive
- `configs/train_smoke.yaml` / `configs/train_gpu.yaml` / `configs/train_mini.yaml` — параметры обучения (`rf_backend: auto|sklearn|cuml` или переменная `IR_RF_BACKEND`)
- `configs/paths.huggingface.yaml` — пути после `fetch-data` без Drive
- `configs/bands_reference.yaml` — полосы (correlation charts + ru.wikipedia), диапазоны см⁻¹, SMARTS

## Лицензии данных

Спектры JCAMP могут содержать указания об авторских правах NIST и др.; перед публикацией проверьте условия использования вашего набора файлов.
