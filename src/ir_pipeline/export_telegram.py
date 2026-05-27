"""Экспорт обученной IrResnet4 в каталог моделей FTIR Telegram-бота."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch

from ir_pipeline.logging_utils import log
from ir_pipeline.models.ir_resnet4 import IrResnet4


def export_irresnet_to_bot(
    run_dir: Path,
    target_models_root: Path,
    *,
    model_version: str | None = None,
) -> Path:
    """
    Создаёт каталог target_models_root/<version>/ с:
      <version>_model_param, <version>_classes.txt, опционально <version>.pt
    """
    bundle_path = run_dir / "irresnet_bundle.pt"
    if not bundle_path.exists():
        raise FileNotFoundError(bundle_path)

    ck = torch.load(bundle_path, map_location="cpu", weights_only=False)
    meta = ck["meta"]
    version = model_version or str(meta.get("model_version", "v0.1.0.34"))
    hidden = int(meta["hidden_size"])
    class_names: list[str] = meta["class_names"]

    out_dir = target_models_root / version
    out_dir.mkdir(parents=True, exist_ok=True)

    model = IrResnet4(hidden_size=hidden, class_nums=len(class_names))
    model.load_state_dict(ck["model_state"])
    model.eval()

    param_path = out_dir / f"{version}_model_param"
    classes_path = out_dir / f"{version}_classes.txt"
    pt_path = out_dir / f"{version}.pt"

    torch.save(model.state_dict(), param_path)
    classes_path.write_text("\n".join(class_names) + "\n", encoding="utf-8")
    torch.save(model, pt_path)

    manifest = {
        "model_version": version,
        "hidden_size": hidden,
        "n_classes": len(class_names),
        "source_run_dir": str(run_dir.resolve()),
        "files": [param_path.name, classes_path.name, pt_path.name],
    }
    (out_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    # copy training curve if present
    for name in ("irresnet_training_curve.png", "irresnet_metrics.json"):
        src = run_dir / name
        if src.is_file():
            shutil.copy2(src, out_dir / name)

    log(f"exported telegram model → {out_dir}")
    return out_dir
