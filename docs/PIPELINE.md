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
| `train` | sklearn/cuML RandomForest |
| `torch-train` | 1D CNN регрессия позиций пиков |
| `irresnet-train` | IrResnet4 multi-label (3 канала, `model_inputs.npz`) |
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

## Colab-ноутбуки

По одному notebook на этап — см. [`notebooks/`](../notebooks/):

- `colab_00_setup.ipynb` — clone, pip
- `colab_01_dataset.ipynb` — fetch + preview
- `colab_02_baseline_rf.ipynb` — RF + метрики
- `colab_03_train_irresnet4.ipynb` — IrResnet4
- `colab_04_gradcam.ipynb` — Grad-CAM

## Условия измерения (газ / раствор / ATR)

В `meta.parquet`: `measurement_mode`, `sample_state`. RandomForest использует one-hot этих полей.
