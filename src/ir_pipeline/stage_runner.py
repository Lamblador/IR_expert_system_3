"""Оркестратор именованных стадий пайплайна."""

from __future__ import annotations

import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ir_pipeline.config_loader import load_yaml, merge_train_defaults, resolve_paths
from ir_pipeline.dataset_build import build_dataset
from ir_pipeline.dataset_telegram import build_telegram_arrays_from_jcamp, plot_dataset_preview, save_telegram_npz
from ir_pipeline.evaluate import evaluate_run
from ir_pipeline.export_telegram import export_irresnet_to_bot
from ir_pipeline.gradcam import run_cam_examples
from ir_pipeline.irresnet_train import train_irresnet_run
from ir_pipeline.logging_utils import configure_log, heartbeat, log
from ir_pipeline.metrics_plot import plot_train_metrics
from ir_pipeline.train_sklearn import train_models


def load_stages_config(path: Path) -> dict[str, Any]:
    return load_yaml(path)


def list_stages(cfg: dict[str, Any]) -> list[str]:
    return list((cfg.get("stages") or {}).keys())


def list_profiles(cfg: dict[str, Any]) -> list[str]:
    return list((cfg.get("profiles") or {}).keys())


def _stage_dir(pipeline_run: Path, stage_key: str, stage_id: str) -> Path:
    d = pipeline_run / f"stage_{stage_id}_{stage_key}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_status(stage_dir: Path, status: str, detail: dict[str, Any] | None = None) -> None:
    payload = {
        "status": status,
        "utc": datetime.now(timezone.utc).isoformat(),
        "detail": detail or {},
    }
    (stage_dir / "stage_status.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run_stage(
    stage_key: str,
    *,
    stages_yaml: Path,
    paths_yaml: Path,
    pipeline_run: Path,
    overrides: dict[str, Any] | None = None,
) -> Path:
    cfg = load_stages_config(stages_yaml)
    defaults = {**(cfg.get("defaults") or {}), **(overrides or {})}
    stage_meta = (cfg.get("stages") or {}).get(stage_key)
    if not stage_meta:
        raise ValueError(f"Неизвестная стадия: {stage_key}. Доступны: {list_stages(cfg)}")

    stage_id = str(stage_meta.get("id", "00"))
    stage_dir = _stage_dir(pipeline_run, stage_key, stage_id)
    configure_log(stage_dir, stage=stage_key)
    log(f"=== stage {stage_id} {stage_key}: {stage_meta.get('description', '')} ===")

    paths_cfg = load_yaml(paths_yaml)
    p = resolve_paths(paths_cfg)
    dv = str(defaults.get("dataset_version", p["dataset_version"]))
    ds_dir = p["processed_root"] / dv
    bands_yaml = p["bands_config"]

    state_path = pipeline_run / "pipeline_state.json"
    state: dict[str, Any] = {}
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))

    try:
        with heartbeat(60.0, f"stage {stage_key} in progress..."):
            if stage_key == "fetch":
                from huggingface_hub import hf_hub_download
                import shutil
                import zipfile

                repo_id = str(defaults.get("hf_repo_id", "Lamblador/IRSpectra2"))
                filename = str(defaults.get("hf_filename", "dataset_mini.zip"))
                extract_to = Path(defaults.get("hf_extract_to", "data/processed"))
                cached = Path(
                    hf_hub_download(
                        repo_id=repo_id,
                        repo_type="dataset",
                        filename=filename,
                        revision="main",
                    )
                )
                out = Path(filename)
                if cached.resolve() != out.resolve():
                    shutil.copy2(cached, out)
                if zipfile.is_zipfile(out):
                    extract_to.mkdir(parents=True, exist_ok=True)
                    with zipfile.ZipFile(out) as zf:
                        zf.extractall(extract_to)
                    log(f"extracted {out} → {extract_to}")

            elif stage_key == "dataset_build":
                max_files = int(defaults.get("max_files", 0))
                build_dataset(
                    raw_jcamp_dir=p["raw_jcamp_dir"],
                    processed_root=p["processed_root"],
                    dataset_version=dv,
                    bands_yaml=bands_yaml,
                    max_files=max_files,
                    resolve_missing_structures=bool(defaults.get("resolve_missing_structures", False)),
                )

            elif stage_key == "dataset_preview":
                if not (ds_dir / "meta.parquet").exists():
                    raise FileNotFoundError(f"Нет датасета {ds_dir}; выполните fetch или dataset_build")
                meta = __import__("pandas").read_parquet(ds_dir / "meta.parquet")
                if not (ds_dir / "telegram_arrays.npz").exists():
                    log("building telegram_arrays.npz")
                    from ir_pipeline.dataset_telegram import build_telegram_arrays_from_npz

                    try:
                        X_bot, tids = build_telegram_arrays_from_jcamp(
                            meta, p["raw_jcamp_dir"], peak_threshold=float(defaults.get("peak_threshold", 0.1))
                        )
                    except Exception:
                        X_bot, tids = build_telegram_arrays_from_npz(
                            ds_dir, peak_threshold=float(defaults.get("peak_threshold", 0.1))
                        )
                    save_telegram_npz(ds_dir, X_bot, tids)
                plots = plot_dataset_preview(ds_dir, stage_dir / "plots", bands_yaml)
                log(f"preview plots: {plots}")

            elif stage_key == "train_rf":
                train_cfg = merge_train_defaults(load_yaml(Path(defaults["train_config"])))
                rd = stage_dir / "rf_run"
                metrics = train_models(
                    dataset_dir=ds_dir,
                    run_dir=rd,
                    mode=str(defaults.get("rf_mode", "spectrum")),
                    train_cfg=train_cfg,
                    random_seed=int(train_cfg["random_seed"]),
                    train_frac=float(train_cfg["train_frac"]),
                )
                state["rf_run_dir"] = str(rd)
                state["rf_metrics"] = metrics

            elif stage_key == "plot_rf_metrics":
                rf_run = Path(state.get("rf_run_dir", stage_dir / "rf_run"))
                if not (rf_run / "metrics.json").exists():
                    raise FileNotFoundError(f"Нет metrics.json в {rf_run}; сначала train_rf")
                p1, p2 = plot_train_metrics(rf_run, bands_yaml, output_dir=stage_dir / "plots")
                log(f"metrics plots: {p1}, {p2}")

            elif stage_key == "train_irresnet":
                if not (ds_dir / "telegram_arrays.npz").exists():
                    from ir_pipeline.dataset_telegram import build_telegram_arrays_from_npz

                    meta = __import__("pandas").read_parquet(ds_dir / "meta.parquet")
                    try:
                        X_bot, tids = build_telegram_arrays_from_jcamp(
                            meta, p["raw_jcamp_dir"], peak_threshold=float(defaults.get("peak_threshold", 0.1))
                        )
                    except Exception:
                        X_bot, tids = build_telegram_arrays_from_npz(
                            ds_dir, peak_threshold=float(defaults.get("peak_threshold", 0.1))
                        )
                    save_telegram_npz(ds_dir, X_bot, tids)
                train_cfg = merge_train_defaults(load_yaml(Path(defaults["train_config_irresnet"])))
                rd = stage_dir / "irresnet_run"
                summary = train_irresnet_run(
                    dataset_dir=ds_dir,
                    run_dir=rd,
                    bands_yaml=bands_yaml,
                    train_cfg=train_cfg,
                    device=defaults.get("device"),
                )
                state["irresnet_run_dir"] = str(rd)
                state["irresnet_summary"] = summary

            elif stage_key == "cam_examples":
                ir_run = Path(state.get("irresnet_run_dir", stage_dir / "irresnet_run"))
                bundle = ir_run / "irresnet_bundle.pt"
                if not bundle.exists():
                    raise FileNotFoundError(f"Нет {bundle}; сначала train_irresnet")
                paths = run_cam_examples(bundle, ds_dir, stage_dir / "cam", n_examples=int(defaults.get("cam_n", 3)))
                state["cam_paths"] = [str(x) for x in paths]

            elif stage_key == "evaluate_rf":
                rf_run = Path(state.get("rf_run_dir", stage_dir / "rf_run"))
                summary = evaluate_run(ds_dir, rf_run, mode=str(defaults.get("rf_mode", "spectrum")))
                (stage_dir / "eval_report.json").write_text(
                    json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
                )

            elif stage_key == "export_telegram":
                ir_run = Path(state.get("irresnet_run_dir", stage_dir / "irresnet_run"))
                target = Path(
                    defaults.get(
                        "telegram_models_dir",
                        r"D:\Programming\Python\FTIR_telegram_bot\models",
                    )
                )
                out = export_irresnet_to_bot(ir_run, target, model_version=defaults.get("export_model_version"))
                state["telegram_export_dir"] = str(out)

            elif stage_key == "predict_smoke":
                rf_run = Path(state.get("rf_run_dir", stage_dir / "rf_run"))
                jcamp = defaults.get("smoke_jcamp")
                if not jcamp:
                    meta = __import__("pandas").read_parquet(ds_dir / "meta.parquet")
                    ok = meta[meta["qc_ok"] == True]  # noqa: E712
                    if ok.empty:
                        raise RuntimeError("Нет qc_ok спектров для smoke predict")
                    jcamp = ok.iloc[0]["path"]
                from ir_pipeline.visualize import predict_file_visualize

                predict_file_visualize(
                    jcamp_path=Path(jcamp),
                    bands_yaml=bands_yaml,
                    bundle_path=rf_run / "models.joblib",
                    out_dir=stage_dir / "predict_out",
                )

            else:
                raise ValueError(f"Стадия не реализована: {stage_key}")

        _write_status(stage_dir, "ok", {"stage": stage_key})
        state["last_stage"] = stage_key
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
        log(f"stage {stage_key} OK → {stage_dir}")
        return stage_dir

    except Exception as e:
        err = traceback.format_exc()
        (stage_dir / "error_log.txt").write_text(err, encoding="utf-8")
        _write_status(stage_dir, "error", {"error": str(e)})
        log(f"stage {stage_key} FAILED: {e}")
        raise


def run_profile(
    profile: str,
    *,
    stages_yaml: Path,
    paths_yaml: Path,
    pipeline_run: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> Path:
    cfg = load_stages_config(stages_yaml)
    profiles = cfg.get("profiles") or {}
    if profile not in profiles:
        raise ValueError(f"Неизвестный профиль {profile}. Доступны: {list_profiles(cfg)}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    pr = pipeline_run or Path("runs") / f"pipeline_{profile}_{ts}"
    pr.mkdir(parents=True, exist_ok=True)
    configure_log(pr, stage="pipeline")
    log(f"profile {profile}: {profiles[profile]}")

    for sk in profiles[profile]:
        run_stage(sk, stages_yaml=stages_yaml, paths_yaml=paths_yaml, pipeline_run=pr, overrides=overrides)

    log(f"profile {profile} complete → {pr}")
    return pr
