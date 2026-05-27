from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ir_pipeline.bands import load_bands


def plot_train_metrics(run_dir: Path, bands_yaml: Path, output_dir: Path | None = None) -> tuple[Path, Path]:
    """Строит PNG: MAE по каждой полосе и среднее MAE по функциональной группе из bands_reference."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    bands_yaml = Path(bands_yaml)
    out = Path(output_dir) if output_dir else run_dir / "plots"
    out.mkdir(parents=True, exist_ok=True)

    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    data = json.loads(metrics_path.read_text(encoding="utf-8-sig"))
    per_band = data.get("per_band_mae") or {}
    if not per_band:
        raise ValueError(f"В {metrics_path} нет ключа per_band_mae")

    band_to_group = {b.band_id: (b.group or "unknown") for b in load_bands(bands_yaml)}

    rows = [(bid, float(mae), band_to_group.get(bid, "unknown")) for bid, mae in sorted(per_band.items(), key=lambda x: x[1])]
    band_ids = [r[0] for r in rows]
    maes = [r[1] for r in rows]
    groups = [r[2] for r in rows]

    uniq_groups = sorted(set(groups))
    cmap = matplotlib.colormaps["tab20"]
    n_g = len(uniq_groups)
    g_to_color = {
        g: cmap(i / max(n_g - 1, 1) if n_g > 1 else 0.0) for i, g in enumerate(uniq_groups)
    }
    colors = [g_to_color[g] for g in groups]

    fig_h = max(6.0, 0.22 * len(rows))
    fig1, ax1 = plt.subplots(figsize=(10, fig_h))
    y_pos = np.arange(len(rows))
    ax1.barh(y_pos, maes, color=colors, height=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(band_ids, fontsize=7)
    ax1.invert_yaxis()
    ax1.set_xlabel("MAE, см⁻¹ (на тестовой части разбиения для полосы)")
    ax1.set_title("Точность по полосам (RandomForest, metrics.json)")
    ax1.grid(axis="x", alpha=0.3)
    fig1.tight_layout()
    p1 = out / "metrics_per_band_mae.png"
    fig1.savefig(p1, dpi=140)
    plt.close(fig1)

    by_group: dict[str, list[float]] = defaultdict(list)
    for _, mae, grp in rows:
        by_group[grp].append(mae)

    g_names = sorted(by_group.keys(), key=lambda g: np.mean(by_group[g]), reverse=True)
    g_mean = [float(np.mean(by_group[g])) for g in g_names]
    g_std = [float(np.std(by_group[g])) if len(by_group[g]) > 1 else 0.0 for g in g_names]
    g_counts = [len(by_group[g]) for g in g_names]

    fig2, ax2 = plt.subplots(figsize=(10, max(4.0, 0.35 * len(g_names))))
    x_pos = np.arange(len(g_names))
    ax2.barh(x_pos, g_mean, xerr=g_std, height=0.65, capsize=3, color="steelblue", alpha=0.85)
    ax2.set_yticks(x_pos)
    labels = [f"{g} (n={c})" for g, c in zip(g_names, g_counts)]
    ax2.set_yticklabels(labels, fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel("Среднее MAE по полосам группы, см⁻¹")
    ax2.set_title("Зависимость ошибки от функциональной группы (справочник полос)")
    ax2.grid(axis="x", alpha=0.3)
    fig2.tight_layout()
    p2 = out / "metrics_by_group_mae.png"
    fig2.savefig(p2, dpi=140)
    plt.close(fig2)

    return p1, p2
