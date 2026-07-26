# API для ноутбуков

Функции и CLI, которые вызываются из `notebooks/colab_*.ipynb`. Полный пакет шире — здесь только notebook-facing поверхность.

## Конфигурация путей

### `load_yaml(path)` — `ir_pipeline.config_loader`

Читает YAML в `dict`.  
**Ноутбуки:** ячейка C, обучение IrResnet/KAN.

### `resolve_paths(paths_cfg)` — `ir_pipeline.config_loader`

Из конфига/`IR_*` env собирает:

| Ключ | Смысл |
|------|--------|
| `raw_jcamp_dir` | каталог JCAMP |
| `processed_root` | родитель версий датасета |
| `dataset_version` | имя каталога (`dataset_v003`, …) |
| `bands_config` | путь к `bands_reference.yaml` |
| `dataset_profile` | `auto` / `mini` / `full` |

**Пишет на диск:** нет.  
**Ноутбуки:** все после ячейки C.

### `merge_train_defaults(train_cfg)` — `ir_pipeline.config_loader`

Дополняет train-yaml значениями по умолчанию (epochs, lr, …).  
**Ноутбуки:** 03, 05, 06, 07.

---

## Датасет и превью

### `plot_dataset_preview(dataset_dir, out_dir, bands_yaml)` — `ir_pipeline.dataset_preview`

Строит PNG с примерами спектров и меток spectrum/structure.  
**Пишет:** `out_dir/preview_*.png`.  
**Ноутбуки:** 01.

### `build_multilabel_matrix(dataset_dir, spec_ids, bands_yaml, label_schema=…)` — `ir_pipeline.dataset_preview`

Матрица multi-label `Y` и имена классов по схеме `spectrum` | `structure` | `structure_smarts`.  
**Ноутбуки:** 05, 06, 07.

### `load_model_inputs(dataset_dir)` — `ir_pipeline.resnet_input`

Читает `model_inputs.npz`: тензор `(N, 3, 1801)`, wavenumbers, spectrum_id, контекст.  
**Ноутбуки:** 05–07.

### `load_split_ids(dataset_dir)` — `ir_pipeline.dataset_split`

`train` / `val` / `test` id из `split.json`.  
**Ноутбуки:** 07.

---

## Обучение IrResnet / KAN

### `train_irresnet_run(dataset_dir, run_dir, bands_yaml, train_cfg, …)` — `ir_pipeline.irresnet_train`

Обучение multi-label модели (`model_family` в cfg: `irresnet4`, `kan_hybrid`, `kan_full`).  
**Пишет:** `run_dir/irresnet_bundle.pt`, `irresnet_history.json`, метрики, кривые.  
**Ноутбуки:** 03, 05, 07.

### `IrResnetTrainer` / `IrDataset` — `ir_pipeline.irresnet_train`

Тренер (этап 06) и `torch.utils.data.Dataset` для батчей спектр+контекст+метки.  
**Ноутбуки:** 06, 07.

### `build_spectrum_model(family, …)` — `ir_pipeline.models.model_factory`

Фабрика M0/M1/M2.  
**Ноутбуки:** 07 (инференс / загрузка bundle).

---

## Grad-CAM

### `compute_cam(model, x, class_idx)` / `interpolate_cam_to_wavenumbers` — `ir_pipeline.gradcam`

Карта важности → ось см⁻¹.  
**Ноутбуки:** 07 (визуализация); этап 04 использует CLI.

---

## CLI (через `!ir-pipeline` или subprocess)

| Команда | Назначение | Ноутбук |
|---------|------------|---------|
| `fetch-data --filename … --extract-to …` | скачать zip с HF | C (smoke), 00 |
| `train --paths … --run-dir …` | RandomForest | 02 |
| `plot-train-metrics --run-dir …` | MAE PNG | 02 |
| `irresnet-train …` | IrResnet через CLI | 04 (fallback) |
| `gradcam-examples --run-dir …` | Grad-CAM PNG | 04 |
| `dataset-split --dataset-version …` | split v2 70/10/20 | 06 |

Всегда передавайте `--paths` из `PATHS_YAML` шапки.

---

## Tools (вне ноутбуков, для данных)

### `tools/build_sdbs_holdout.py`

Hold-out SDBS вне `dataset_v003` → `data/external_sdbs/sdbs_eval.npz` (+ meta, manifest).  
Аргументы: `--raw-jcamp`, `--dataset-v003`, `--out-dir`, `--max-files`.  
**Ноутбуки:** 07 читает готовый npz.

### `tools/view_dataset_samples.py`

Визуальный QC: спектр + SMILES (RDKit) → PNG в `--out-dir`.  
Аргументы: `--dataset-dir`, `--random-k` / `--ids` / `--indices`.

---

## Переменные окружения

| Env | Эффект |
|-----|--------|
| `IR_RAW_JCAMP_DIR` | перекрывает `raw_jcamp_dir` |
| `IR_PROCESSED_ROOT` | перекрывает `processed_root` (Colab Drive выставляет в C) |
| `IR_DATASET_PROFILE` | `mini` / `full` / `auto` |
| `HF_TOKEN` | private HF repo |
| `IR_RF_BACKEND` | `auto` / `sklearn` / `cuml` |
