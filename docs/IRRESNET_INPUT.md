# Вход IrResnet4 (400–4000 см⁻¹ + контекст измерения)

## Сетка

Совпадает с датасетом: **400–4000 см⁻¹**, шаг **2** → **1801** точка.

Тензор на спектр `(3, 1801)` в `model_inputs.npz`:

| Канал | Содержимое |
|-------|------------|
| 0 | ось wavenumber (фиксированная сетка) |
| 1 | поглощение (`X_absorbance_corrected` на сетке датасета) |
| 2 | маска пиков (peakutils) |

После смены сетки с 500–4100 удалите старый `model_inputs.npz` или пересоберите датасет — файл пересоздаётся автоматически при несовпадении `version`.

## Контекст условий измерения

**Проблема:** ATR, газовая ячейка, раствор (KBr / CHCl₃) дают разную форму полос, сдвиги ν и интенсивности.

**Реализация (как у RandomForest, но для CNN):** вектор one-hot из `meta.parquet` конкатенируется к признакам **после** свёрточного encoder, перед полносвязными слоями.

Категории (`measurement_context.py`):

- **measurement_mode:** `atr`, `transmission`, `gas`, `solution`, `absorbance`, `unknown`
- **sample_state:** `solid`, `liquid`, `gas`, `solution`, `film`, `unknown`

Итого **12** бинарных признаков (по одному активному в каждой группе). Включение: `use_measurement_context: true` в `configs/train_irresnet.yaml`.

## Альтернативы (не реализованы)

| Подход | Плюсы | Минусы |
|--------|-------|--------|
| Доп. каналы (broadcast one-hot по L) | Контекст «видит» каждая позиция по оси ν | Увеличивает вход conv; Grad-CAM сложнее |
| FiLM (γ, β от embedding) | Гибкая модуляция по слоям | Больше кода, нужен tuning |
| Embedding + attention | Компактно | Сложнее интерпретировать |

Текущий вариант — минимальное изменение архитектуры и прямая связь с полями JCAMP.

## Команды

```bash
# пересоздать model_inputs.npz (новая сетка + контекст)
ir-pipeline irresnet-train --paths configs/paths.local.yaml --dataset-version dataset_v001
# по умолчанию --label-schema structure (SMARTS); спектральные метки: --label-schema spectrum

ir-pipeline gradcam-examples --run-dir runs/<irresnet_run> --paths configs/paths.local.yaml
```

Отключить контекст: в yaml `use_measurement_context: false` или переобучить без `X_context`.

## Гиперпараметры и мониторинг (Colab)

Файлы: `configs/train_irresnet_original.yaml` (full/v003), `configs/train_irresnet_mini.yaml` (smoke), `configs/train_irresnet_colab.yaml` (live-графики).

| Ключ | По умолчанию (original) | Описание |
|------|------------------------|----------|
| `label_schema` | `structure_smarts` | SMARTS-only (как оригинал) |
| `torch_epochs` | 200 | upper bound; early stopping |
| `torch_batch_size` | 32 | размер батча |
| `torch_lr` | 1e-5 | learning rate |
| `torch_scheduler` | `steplr` | StepLR(75, γ=0.2) |
| `ir_hidden_size` | 72 | full; mini: 34 |
| `use_weighted_sampler` | true | WeightedRandomSampler |
| `context_dropout_prob` | 0.25 | unknown context при train |
| `early_stop_metric` | `val_lrap` | чекпоинт по val |
| `augment_train` | true | online-аугментация спектра |
**Датасет:** `--dataset-profile auto|mini|full` — см. [`data/README.md`](../data/README.md).

**Live-график в Colab:** `IrResnetTrainer` / `train_irresnet_run()` (см. `colab_03`, `colab_05`, `colab_06`). CLI в subprocess **не** обновляет график в ноутбуке.

**Схемы меток (`--label-schema`):**

| Значение | Файл | Позитив |
|----------|------|---------|
| `spectrum` | `labels_spectrum.parquet` | пик в регионе |
| `structure` | `labels_structure.parquet` | SMARTS + пик |
| `structure_smarts` | `labels_structure_smarts.parquet` | SMARTS (без требования пика) |

1D CNN (`torch-train`): `configs/train_torch_colab.yaml`, loss — `smooth_l1` или `mse`.

```bash
ir-pipeline irresnet-train --dataset-profile full --config configs/train_irresnet_original.yaml
ir-pipeline dataset-split --dataset-version dataset_v003
ir-pipeline augment-model-inputs --dataset-version dataset_v003
ir-pipeline dataset-audit-duplicates --dataset-version dataset_v003
```
