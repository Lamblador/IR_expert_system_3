# Ноутбуки: Local Jupyter и Google Colab

Единый dual-workflow. В **каждом** ноутбуке шапка из трёх блоков:

| Блок | Когда запускать | Когда пропускать |
|------|-----------------|------------------|
| **A. Colab** | Google Colab | локальный Jupyter |
| **A2. Drive** | Colab + полный датасет на Drive | HF smoke, локально |
| **B. Local** | локальный Jupyter | Colab |
| **C. Пути и данные** | всегда после A или B | — |

Контракт после **C**: `ROOT`, `PATHS_YAML`, `paths`, `DATASET_DIR`, `BANDS_YAML`, `RUNS_DIR`.

Источник истины для `colab_00`…`05` и `07`: [`notebooks/_make_colab_notebooks.py`](../notebooks/_make_colab_notebooks.py).  
После правок генератора: `python notebooks/_make_colab_notebooks.py`.  
`colab_06` и `colab_launcher` — вручную / [`_sync_manual_notebooks.py`](../notebooks/_sync_manual_notebooks.py).

## Режимы данных (ячейка C)

| `DATA_MODE` | YAML | Датасет |
|-------------|------|---------|
| `local` | `configs/paths.local.yaml` | версия из yaml (часто `dataset_v003`) |
| `colab_smoke` | `configs/paths.huggingface.yaml` | `dataset_mini` (+ HF fetch) |
| `colab_full` | `configs/paths.colab.yaml` | Drive `…/ir_data/processed/dataset_v003` |
| `auto` | по `IR_ENV` и наличию `IR_DATA` | см. код ячейки C |

Локально: скопируйте [`configs/paths.local.example.yaml`](../configs/paths.local.example.yaml) → `paths.local.yaml` и пропишите диски.

## Оглавление

| № | Файл | Операции | Артефакты |
|---|------|----------|-----------|
| — | [`colab_launcher.ipynb`](../notebooks/colab_launcher.ipynb) | навигация | — |
| 0 | [`colab_00_setup.ipynb`](../notebooks/colab_00_setup.ipynb) | env + проверка CLI/датасета | установленный пакет |
| 1 | [`colab_01_dataset.ipynb`](../notebooks/colab_01_dataset.ipynb) | `plot_dataset_preview` | `runs/colab_preview/plots/` |
| 2 | [`colab_02_baseline_rf.ipynb`](../notebooks/colab_02_baseline_rf.ipynb) | `ir-pipeline train` + MAE | `runs/colab_pipeline_rf/` |
| 3 | [`colab_03_train_irresnet4.ipynb`](../notebooks/colab_03_train_irresnet4.ipynb) | `train_irresnet_run` | `irresnet_bundle.pt` |
| 4 | [`colab_04_gradcam.ipynb`](../notebooks/colab_04_gradcam.ipynb) | `gradcam-examples` | `reports/colab_gradcam/` |
| 5 | [`colab_05_irresnet_experiments.ipynb`](../notebooks/colab_05_irresnet_experiments.ipynb) | E1–E4 | `runs/exp_v003/` |
| 6 | [`colab_06_irresnet_train_drive.ipynb`](../notebooks/colab_06_irresnet_train_drive.ipynb) | A/B v002 original protocol | `runs/colab06_*` |
| 7 | [`colab_07_kan_compare.ipynb`](../notebooks/colab_07_kan_compare.ipynb) | M0/M1/M2 + SDBS | `runs/kan_cmp_*` |

API функций: [`NOTEBOOK_API.md`](NOTEBOOK_API.md).

---

## Этап 0 — setup

1. Markdown: цель этапа.
2. **A** или **B** (не оба).
3. **A2** только для Drive full.
4. **C** — пути.
5. Опционально: zip-распаковка (Colab).
6. Проверка `ir-pipeline --help` и `spectra.npz`.

## Этап 1 — датасет

После шапки: `plot_dataset_preview(DATASET_DIR, …)` → PNG превью меток.

## Этап 2 — RandomForest

`ir-pipeline train --paths {PATHS_YAML} --dataset-version …` → `plot-train-metrics`.  
Скачивание zip модели — **только Colab** (ячейка с `google.colab.files`).

## Этап 3 — IrResnet4

`train_irresnet_run(...)` с `DATASET_DIR` из C. Конфиг: `train_irresnet_colab.yaml`.

## Этап 4 — Grad-CAM

Ищет `irresnet_bundle.pt` в `runs/`; если нет — обучает на текущем датасете.  
Ручные индексы: раскомментировать строку с `--spectrum-indices`.

## Этап 5 — эксперименты E1–E4

Нужны `labels_structure_smarts.parquet` (лучше v003). Пишет `runs/exp_v003/summary.json`.

## Этап 6 — A/B v002

После C — ячейка **фиксации `dataset_v002`**. Split v2 через `dataset-split`.  
Local: под `processed_root` должен лежать `dataset_v002/`.

## Этап 7 — KAN compare

Нужен `model_inputs.npz` в `dataset_v003`. SDBS: `data/external_sdbs/sdbs_eval.npz` или Drive `ir_data/external_sdbs/`.  
Сборка holdout: `python tools/build_sdbs_holdout.py`.

---

## Типичные ошибки

| Симптом | Причина | Что сделать |
|---------|---------|-------------|
| `No module named google.colab` | запущена ячейка A/A2 локально | выполнить только **B** |
| Nested `notebooks/IR_expert_system_3/` | старый bootstrap | ячейка **B** ищет корень walk-up; вложенный клон можно удалить |
| `Нет spectra.npz` | неверный yaml / нет Drive / нет fetch | проверить C / `paths.local.yaml` / A2 |
| `Нет paths.local.yaml` | не скопирован example | `copy configs/paths.local.example.yaml configs/paths.local.yaml` |
| Рассинхрон кода Colab | ветка `colab-v1` | A делает pull; локально ветку не трогает |

## Регенерация

```bash
python notebooks/_make_colab_notebooks.py
python notebooks/_sync_manual_notebooks.py   # 06 + launcher
```
