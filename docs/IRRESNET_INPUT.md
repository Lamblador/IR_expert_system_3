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

ir-pipeline gradcam-examples --run-dir runs/<irresnet_run> --paths configs/paths.local.yaml
```

Отключить контекст: в yaml `use_measurement_context: false` или переобучить без `X_context`.
