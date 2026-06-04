# Каталог датасетов (`data/processed/`)

## Версии

| Версия | Назначение | Split | Метки |
|--------|------------|-------|-------|
| `dataset_mini` | Colab / smoke (~300 JCAMP) | v1 (85/15) или v2 после `dataset-split` | 3 parquet |
| `dataset_v001` | Полный baseline HF | v1 (85/15) | 3 parquet |
| `dataset_v002` | IrResnet exp1 | v1 (85/15) | + `labels_structure_smarts` |
| **`dataset_v003`** | **Production train (full)** | **v2: 70/10/20**, stratified SMARTS | как v002 |

## Выбор mini / full

В `configs/paths.*.yaml`:

- `dataset_profile: auto` — если есть `dataset_v003/spectra.npz` → full, иначе `dataset_mini`
- `dataset_profile: mini` / `full`
- env: `IR_DATASET_PROFILE=mini|full|auto`

CLI: `ir-pipeline irresnet-train --dataset-profile full`

## Файлы в каталоге версии

- `spectra.npz` — поглощение, сетка 400–4000 см⁻¹
- `meta.parquet` — QC, SMILES, режим измерения
- `labels_spectrum.parquet` — пики в регионах
- `labels_structure.parquet` — SMARTS + пик
- `labels_structure_smarts.parquet` — **SMARTS-only** (дефолт IrResnet)
- `split.json` — v1: train/test; v2: train/val/test
- `split_report.json` — покрытие полос по fold (v2)
- `model_inputs.npz` — 3-канальный вход CNN
- `model_inputs_aug.npz` — offline-аугментация train (опционально)
- `manifest.json` — статистика сборки

## Подготовка `dataset_v003`

```bash
ir-pipeline dataset-copy-version --from-version dataset_v002 --to-version dataset_v003
ir-pipeline dataset-split --dataset-version dataset_v003 --label-schema structure_smarts
ir-pipeline dataset-split-audit --dataset-version dataset_v003
```

## Схемы меток (IrResnet)

| `label_schema` | Позитив |
|----------------|---------|
| `structure_smarts` | SMARTS совпал (как оригинал 72 cls) |
| `structure` | SMARTS + пик в регионе |
| `spectrum` | пик без SMARTS |

## Аугментация и контекст

- **Контекст измерения:** 12 one-hot; при обучении `context_dropout_prob` → «unknown»
- **Online:** `augment_train: true` в yaml
- **Offline:** `ir-pipeline augment-model-inputs --dataset-version dataset_v003`

## Hugging Face zip

- `dataset_mini.zip` — smoke
- `dataset_v001.zip` — полный корпус
- `dataset_v003.zip` — после локальной подготовки split v2

## Сравнение с оригиналом (v1.5.1.72)

Оригинал: ~14k NIST, 72 SMARTS-класса, split 70/10/20, hidden=72, lr=1e-5, WRS + pos_weight, LRAP.

v3: 61 полоса `bands_reference.yaml`, `structure_smarts`, тот же протокол в `configs/train_irresnet_original.yaml`.
