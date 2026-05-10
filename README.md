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

Если Google Drive не подходит, загрузите данные в Hugging Face Dataset repo и скачивайте их командой:

```bash
# Raw JCAMP: архив должен распаковаться в ./downloaded_jcamp/
ir-pipeline fetch-data --repo-id USER/ir-expert-data --filename downloaded_jcamp.zip --extract-to .

# Уже собранный датасет: архив должен распаковаться в ./data/processed/dataset_v001/
ir-pipeline fetch-data --repo-id USER/ir-expert-data --filename dataset_v001.zip --extract-to data/processed
```

Для private repo задайте токен в окружении:

```bash
export HF_TOKEN=hf_...
# Windows PowerShell:
$env:HF_TOKEN = "hf_..."
```

Что загрузить на Hugging Face:

- `downloaded_jcamp.zip` — исходники для полной пересборки; внутри должен быть каталог `downloaded_jcamp/` с JCAMP-файлами (`*.jdx` или файлы без расширения с CAS-именами).
- `dataset_v001.zip` — быстрый вариант для обучения без raw-файлов; внутри должен быть каталог `dataset_v001/` с `spectra.npz`, `meta.parquet`, `labels_spectrum.parquet`, `labels_structure.parquet`, `unresolved_structures.parquet`, `structure_cache.parquet`, `split.json`, `manifest.json`.
- Опционально `lamblador_irspectra_structures.parquet` — seed-кэш структур из Lamblador; если не загрузить, пайплайн сам пересоберёт его из публичного источника.

После `build-dataset` в `data/processed/<dataset_version>/` появляются:

- `spectra.npz`: `X` (нормализованный спектр), `X_absorbance_corrected`, `X_absorbance_like_interp`, `coverage`, `wavenumbers`
- `meta.parquet`, `labels_spectrum.parquet`, `labels_structure.parquet`, `unresolved_structures.parquet`
- `structure_cache.parquet`, `split.json` (фиксированное разбиение для обучения/валидации), `manifest.json`

## Команды

```bash
# Быстрая сборка версионированного датасета: только локальный/HF seed-кэш, без PubChem
ir-pipeline build-dataset --paths configs/paths.local.yaml --max-files 0

# Опционально: медленно добрать структуры, которых нет в быстром кэше, затем пересобрать датасет
ir-pipeline resolve-missing-structures --paths configs/paths.local.yaml --dataset-version dataset_v001
ir-pipeline build-dataset --paths configs/paths.local.yaml --max-files 0

# Быстрая проверка на подвыборке
ir-pipeline build-dataset --paths configs/paths.local.yaml --max-files 100 --dataset-version dataset_smoke

# Скачать данные из Hugging Face Dataset repo без Google Drive
ir-pipeline fetch-data --repo-id USER/ir-expert-data --filename downloaded_jcamp.zip --extract-to .

# Обучение RandomForest (режим только спектр или спектр + SMARTS-маска)
ir-pipeline train --paths configs/paths.local.yaml --dataset-version dataset_v001 --mode spectrum --config configs/train_smoke.yaml
ir-pipeline train --paths configs/paths.local.yaml --dataset-version dataset_v001 --mode spectrum_structure --config configs/train_smoke.yaml

# Оценка
ir-pipeline evaluate --paths configs/paths.local.yaml --dataset-version dataset_v001 --run-dir runs/run_001

# Предсказание + PNG + текстовое описание
ir-pipeline predict --paths configs/paths.local.yaml --jcamp path/to/file --run-dir runs/run_001 --output-dir reports/out

# PyTorch: многозадачная 1D CNN по позициям пиков (после pip install -e ".[torch]")
ir-pipeline torch-train --paths configs/paths.local.yaml --dataset-version dataset_smoke --mode spectrum --config configs/train_smoke.yaml
# → torch_bundle.pt, torch_training_curve.png, torch_history.json

ir-pipeline predict ... --run-dir runs/<torch_run> --torch --output-dir reports/out_torch
```

Справочник [`configs/bands_reference.yaml`](configs/bands_reference.yaml) дополнен строками по таблице Википедии «[Таблица характеристических частот в инфракрасной спектроскопии](https://ru.wikipedia.org/wiki/Таблица_характеристических_частот_в_инфракрасной_спектроскопии)» (`source: ru.wikipedia IR table`); скорректированы интервалы для `ch_sp3`, замечания для `nh_stretch`.

## Конфигурация

- `configs/paths.local.yaml` — локальные пути
- `configs/paths.colab.yaml` — пример для Google Drive
- `configs/train_smoke.yaml` / `configs/train_gpu.yaml` — параметры обучения
- `configs/bands_reference.yaml` — полосы (correlation charts + ru.wikipedia), диапазоны см⁻¹, SMARTS

## Лицензии данных

Спектры JCAMP могут содержать указания об авторских правах NIST и др.; перед публикацией проверьте условия использования вашего набора файлов.
