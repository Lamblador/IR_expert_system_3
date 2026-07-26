# Пайплайн IR Expert System — этапы

Пайплайн разбит на **именованные стадии**. Оркестратор: `ir-pipeline run`.

## Быстрый старт (Colab / HF mini-dataset)

```bash
pip install -e ".[torch]"
ir-pipeline run profile smoke --paths configs/paths.huggingface.yaml
```

Стадии профиля `smoke` (см. [`configs/stages.yaml`](../configs/stages.yaml)):

| № | Стадия | Что делает | Артефакты |
|---|--------|------------|-----------|
| 01 | `fetch` | Скачивание `dataset_mini.zip` с Hugging Face | `data/processed/dataset_mini/` |
| 03 | `dataset_preview` | Превью спектров (spectrum/structure labels) | `runs/.../plots/*.png` |
| 04 | `train_rf` | RandomForest по полосам | `rf_run/models.joblib`, `metrics.json` |
| 05 | `plot_rf_metrics` | MAE по полосам и группам | `plots/metrics_*.png` |

## Локальный полный цикл

```bash
ir-pipeline build-mini-dataset --paths configs/paths.local.yaml
ir-pipeline run profile full_local --paths configs/paths.local.yaml
```

## Одна стадия

```bash
ir-pipeline run stage dataset_preview --paths configs/paths.local.yaml --dataset-version dataset_mini
ir-pipeline run list
```

## Отдельные команды (без оркестратора)

| Команда | Назначение |
|---------|------------|
| `build-dataset` / `build-mini-dataset` | JCAMP → `spectra.npz`, labels |
| `train` | sklearn/cuML RandomForest (**по умолчанию** `spectrum_structure` → SMARTS) |
| `torch-train` | 1D CNN (**по умолчанию** `spectrum_structure` → `labels_structure.parquet`) |
| `irresnet-train` | IrResnet4 multi-label (`--label-schema structure` \| `structure_smarts` \| `spectrum`) |
| `dataset-audit-duplicates` | Отчёт о дублях (без удаления) |
| `gradcam-examples` | Grad-CAM PNG (авто или `--spectrum-indices`, `--class-indices`) |
| `predict` | Инференс по JCAMP + PNG |

## Grad-CAM вручную

```bash
ir-pipeline irresnet-train --paths configs/paths.local.yaml --dataset-version dataset_mini
ir-pipeline gradcam-examples --run-dir runs/<irresnet_run> --n-examples 3
ir-pipeline gradcam-examples --run-dir runs/<irresnet_run> \
  --spectrum-indices 0,5,12 --class-indices 3,12 --output-dir reports/gradcam
```

## Логи и «не зависло»

Каждая стадия пишет в:

- `runs/<pipeline_run>/stage_XX_<name>/pipeline.log`
- консоль с префиксом `[ir-pipeline][stage_name]`
- heartbeat каждые ~60 с на долгих шагах

При ошибке: `error_log.txt` и `stage_status.json` в каталоге стадии.

## Ноутбуки (Local + Colab)

Dual-workflow (ячейки **A** Colab / **B** Local / **C** пути): см. [`NOTEBOOKS.md`](NOTEBOOKS.md), API — [`NOTEBOOK_API.md`](NOTEBOOK_API.md).

| Ноутбук | Стадии / команды | Артефакты |
|---------|------------------|-----------|
| `colab_00_setup` | env + проверка данных | установленный пакет |
| `colab_01_dataset` | `dataset_preview` | preview PNG |
| `colab_02_baseline_rf` | `train_rf`, `plot_rf_metrics` | RF run + MAE |
| `colab_03_train_irresnet4` | `irresnet-train` | `irresnet_bundle.pt` |
| `colab_04_gradcam` | `gradcam-examples` | Grad-CAM PNG |
| `colab_05_irresnet_experiments` | E1–E4 `train_irresnet_run` | `runs/exp_v003/` |
| `colab_06_irresnet_train_drive` | A/B v002 original protocol | `runs/colab06_*` |
| `colab_07_kan_compare` | M0/M1/M2 + SDBS | `runs/kan_cmp_*` |

Оглавление: [`notebooks/colab_launcher.ipynb`](../notebooks/colab_launcher.ipynb).

## Условия измерения (газ / раствор / ATR)

В `meta.parquet`: `measurement_mode`, `sample_state`. RandomForest использует one-hot этих полей.
