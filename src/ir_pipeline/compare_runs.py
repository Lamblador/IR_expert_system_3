"""Сводная таблица метрик нескольких run-dir (M0/M1/M2)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_run_metrics(run_dir: Path) -> dict:
    path = run_dir / "irresnet_metrics.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["run_dir"] = str(run_dir)
    data["run_name"] = run_dir.name
    return data


def compare_runs(run_dirs: list[Path]) -> pd.DataFrame:
    rows = [load_run_metrics(d) for d in run_dirs]
    cols = [
        "run_name",
        "model_family",
        "n_params",
        "best_epoch",
        "train_wall_time_sec",
        "test_f1_weighted",
        "test_f1_macro",
        "test_lrap",
        "prediction_threshold",
        "n_train",
        "n_test",
    ]
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols]


def main() -> None:
    ap = argparse.ArgumentParser(description="Сравнение IrResnet/KAN runs")
    ap.add_argument("run_dirs", nargs="+", type=Path, help="каталоги runs с irresnet_metrics.json")
    ap.add_argument("-o", "--output", type=Path, default=None, help="CSV или JSON путь")
    args = ap.parse_args()
    df = compare_runs(args.run_dirs)
    print(df.to_string(index=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.suffix.lower() == ".json":
            args.output.write_text(df.to_json(orient="records", indent=2), encoding="utf-8")
        else:
            df.to_csv(args.output, index=False)
        print(f"saved → {args.output}")


if __name__ == "__main__":
    main()
