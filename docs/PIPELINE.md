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
| 03 | `dataset_preview` | Превью спектров, баланс классов | `runs/.../stage_03_.../plots/*.png` |
| 04 | `train_rf` | RandomForest по полосам | `stage_04_.../rf_run/models.joblib`, `metrics.json` |
| 05 | `plot_rf_metrics` | MAE по полосам и группам | `plots/metrics_*.png` |
| 06 | `train_irresnet` | IrResnet4 multi-label (3 канала, как в боте) | `irresnet_bundle.pt`, `irresnet_training_curve.png` |
| 07 | `cam_examples` | Grad-CAM | `cam/cam_example_*.png` |
| 09 | `export_telegram` | Экспорт в `FTIR_telegram_bot/models/` | `v0.1.0.34_model_param`, `*_classes.txt` |

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
| `build-dataset` / `build-mini-dataset` | JCAMP → `spectra.npz`, labels, **`telegram_arrays.npz`** |
| `train` | sklearn/cuML RandomForest |
| `torch-train` | 1D CNN регрессия позиций пиков (экспериментальная колея) |
| `irresnet-train` | IrResnet4 multi-label для Telegram/CAM |
| `cam-examples` | Grad-CAM картинки |
| `export-telegram` | Копирование весов в бот |
| `predict` | Инференс по JCAMP + PNG |

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
- `colab_04_cam_examples.ipynb` — CAM
- `colab_05_export_telegram.ipynb` — экспорт в бот

## Условия измерения (газ / раствор / ATR)

В `meta.parquet`: `measurement_mode`, `sample_state`. RandomForest уже использует one-hot этих полей. Для IrResnet4 каналы совпадают с ботом; доменные различия учитываются через разнообразие обучающей выборки.
