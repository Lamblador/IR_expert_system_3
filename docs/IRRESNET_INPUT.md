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

Файлы: `configs/train_irresnet.yaml` (локально), `configs/train_irresnet_colab.yaml` (live-графики).

| Ключ | По умолчанию | Описание |
|------|--------------|----------|
| `torch_epochs` | 30 | число эпох |
| `torch_batch_size` | 32 | размер батча |
| `torch_lr` | 0.001 | learning rate |
| `torch_optimizer` | `adamw` | `adamw`, `adam`, `sgd` |
| `torch_weight_decay` | 1e-4 | L2 для AdamW/Adam/SGD |
| `torch_loss` | `bce_with_logits` | multi-label loss |
| `ir_hidden_size` | 34 | ширина FC в IrResnet4 |
| `pos_weight_scale` | 1.0 | масштаб pos_weight в BCE |
| `live_training_plot` | `true` в colab yaml | каждую эпоху: лог последних 5 значений, clear_output, график |
| `train_log_tail` | 5 | сколько последних эпох печатать в лог |

1D CNN (`torch-train`): `configs/train_torch_colab.yaml`, loss — `smooth_l1` или `mse`.

```bash
ir-pipeline irresnet-train --config configs/train_irresnet_colab.yaml --run-dir runs/my_run ...
```
