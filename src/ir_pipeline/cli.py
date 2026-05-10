from __future__ import annotations

import os
import shutil
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath

import click
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError, RepositoryNotFoundError
from tqdm import tqdm

from ir_pipeline.config_loader import load_yaml, merge_train_defaults, resolve_paths
from ir_pipeline.dataset_build import build_dataset, resolve_missing_structures_for_dataset
from ir_pipeline.evaluate import evaluate_run
from ir_pipeline.train_sklearn import train_models
from ir_pipeline.visualize import predict_file_visualize
from ir_pipeline import torch_train as torch_train_mod


@click.group()
def main():
    """IR Pipeline CLI: сборка датасета, обучение, оценка, предсказание."""


def _download_with_progress(url: str, output: Path, token: str | None = None) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "ir-pipeline/0.1"}
    if token:
        headers["Authorization"] = f"Bearer {token.strip()}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as resp:
        total_raw = resp.headers.get("Content-Length")
        total = int(total_raw) if total_raw else None
        with output.open("wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=f"Download {output.name}",
        ) as bar:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                bar.update(len(chunk))
    return output


@main.command("fetch-data")
@click.option(
    "--repo-id",
    default="Lamblador/IRSpectra2",
    show_default=True,
    help="Hugging Face Dataset repo",
)
@click.option("--filename", default="downloaded_jcamp.zip", show_default=True, help="файл в repo")
@click.option("--revision", default="main", show_default=True)
@click.option("--output", type=click.Path(path_type=Path), default=None, help="куда сохранить файл")
@click.option("--extract-to", type=click.Path(path_type=Path), default=None, help="куда распаковать zip после скачивания")
@click.option("--token-env", default="HF_TOKEN", show_default=True, help="env var с HF token для private repo")
@click.option(
    "--token",
    default=None,
    help="HF token напрямую (нежелательно в истории shell); предпочтительнее HF_TOKEN",
)
@click.option(
    "--legacy-http",
    is_flag=True,
    help="скачивать через urllib (без LFS); только если huggingface_hub даёт сбой",
)
def fetch_data_cmd(
    repo_id: str,
    filename: str,
    revision: str,
    output: Path | None,
    extract_to: Path | None,
    token_env: str,
    token: str | None,
    legacy_http: bool,
):
    """Скачать данные из Hugging Face Dataset repo (LFS через huggingface_hub)."""
    out = Path(output) if output else Path(PurePosixPath(filename).name)
    out.parent.mkdir(parents=True, exist_ok=True)
    token_val = (token or os.environ.get(token_env) or "").strip() or None

    hub_page = f"https://huggingface.co/datasets/{repo_id}/tree/{revision}"

    if legacy_http:
        filename_url = urllib.parse.quote(filename, safe="/")
        revision_url = urllib.parse.quote(revision, safe="")
        url = f"https://huggingface.co/datasets/{repo_id}/resolve/{revision_url}/{filename_url}"
        click.echo(f"Downloading (legacy HTTP): {url}")
        try:
            downloaded = _download_with_progress(url, out, token=token_val)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise click.ClickException(
                    f"Hugging Face вернул HTTP {e.code}: для private Dataset нужен токен с правом Read.\n"
                    "  • Задайте: export HF_TOKEN=hf_...\n"
                    "  • Либо: ir-pipeline fetch-data ... --token hf_...\n"
                    "  • Убедитесь, что аккаунт токена добавлен в collaborators этого Dataset repo.\n"
                    "  • Проверьте имя файла и ветку (--revision).\n"
                    "  • Для LFS-файлов не используйте --legacy-http."
                ) from e
            raise click.ClickException(f"HTTP {e.code}: {e.reason}") from e
    else:
        click.echo(f"Downloading via huggingface_hub: datasets/{repo_id} @ {revision} → {filename}")
        try:
            cached_path = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    filename=filename,
                    revision=revision,
                    token=token_val if token_val else None,
                )
            )
        except RepositoryNotFoundError as e:
            raise click.ClickException(
                f"Репозиторий datasets/{repo_id} не найден или недоступен: {e}\n"
                f"  • Откройте {hub_page} и проверьте имя и видимость (Public)."
            ) from e
        except EntryNotFoundError as e:
            raise click.ClickException(
                f"Файл «{filename}» не найден в datasets/{repo_id}: {e}\n"
                f"  • Загрузите архив на вкладке Files или поправьте --filename.\n"
                f"  • Список файлов: {hub_page}"
            ) from e
        except HfHubHTTPError as e:
            code = getattr(getattr(e, "response", None), "status_code", None)
            if code in (401, 403):
                raise click.ClickException(
                    f"Hugging Face вернул HTTP {code}: нужен доступ и токен Read.\n"
                    "  • export HF_TOKEN=hf_... или ir-pipeline fetch-data ... --token hf_...\n"
                    "  • Для gated/private repo: аккаунт токена должен иметь доступ к datasets/{repo_id}.\n"
                    f"  • Карточка репозитория: {hub_page}"
                ) from e
            raise click.ClickException(f"huggingface_hub: {e}") from e
        except Exception as e:
            raise click.ClickException(
                f"{e}\nПодсказка: проверьте имя файла на вкладке Files: {hub_page}"
            ) from e

        if cached_path.resolve() != out.resolve():
            shutil.copy2(cached_path, out)
        downloaded = out
        click.echo(f"Downloaded to {downloaded}")

    if extract_to is not None:
        if not zipfile.is_zipfile(downloaded):
            raise click.ClickException(f"{downloaded} не zip-файл, распаковка невозможна")
        extract_to.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(downloaded) as zf:
            members = zf.infolist()
            with tqdm(total=len(members), desc=f"Extract {downloaded.name}", unit="file") as bar:
                for member in members:
                    zf.extract(member, extract_to)
                    bar.update(1)
        click.echo(f"Extracted to {extract_to}")


@main.command("build-dataset")
@click.option("--paths", type=click.Path(exists=True, path_type=Path), default=Path("configs/paths.local.yaml"))
@click.option("--dataset-version", type=str, default=None, help="override dataset_version из paths yaml")
@click.option("--max-files", type=int, default=0, help="0 = все файлы (осторожно с временем)")
@click.option("--pubchem-sleep", type=float, default=0.12, help="пауза между PubChem запросами в slow-режиме")
@click.option(
    "--resolve-missing-structures",
    is_flag=True,
    help="медленный добор неизвестных CAS/TITLE через PubChem; по умолчанию сборка только по быстрому кэшу",
)
@click.option("--split-seed", type=int, default=42)
@click.option("--train-frac", type=float, default=0.85)
def build_dataset_cmd(
    paths: Path,
    dataset_version: str | None,
    max_files: int,
    pubchem_sleep: float,
    resolve_missing_structures: bool,
    split_seed: int,
    train_frac: float,
):
    cfg = load_yaml(paths)
    p = resolve_paths(cfg)
    dv = dataset_version or str(p["dataset_version"])
    out = build_dataset(
        raw_jcamp_dir=p["raw_jcamp_dir"],
        processed_root=p["processed_root"],
        dataset_version=dv,
        bands_yaml=p["bands_config"],
        max_files=max_files,
        pubchem_sleep_s=pubchem_sleep,
        resolve_missing_structures=resolve_missing_structures,
        split_seed=split_seed,
        train_frac=train_frac,
    )
    click.echo(f"Dataset written to {out}")


@main.command("resolve-missing-structures")
@click.option("--paths", type=click.Path(exists=True, path_type=Path), default=Path("configs/paths.local.yaml"))
@click.option("--dataset-version", type=str, default=None)
@click.option("--pubchem-sleep", type=float, default=0.12, help="пауза между PubChem запросами")
def resolve_missing_structures_cmd(paths: Path, dataset_version: str | None, pubchem_sleep: float):
    """Медленный второй поток: добрать неизвестные структуры и прогреть structure_cache."""
    cfg = load_yaml(paths)
    p = resolve_paths(cfg)
    dv = dataset_version or str(p["dataset_version"])
    report = resolve_missing_structures_for_dataset(
        processed_root=p["processed_root"],
        dataset_version=dv,
        pubchem_sleep_s=pubchem_sleep,
    )
    click.echo(f"Resolved missing structures: {report}")


@main.command("train")
@click.option("--paths", type=click.Path(exists=True, path_type=Path), default=Path("configs/paths.local.yaml"))
@click.option("--dataset-version", type=str, default=None)
@click.option("--config", type=click.Path(exists=True, path_type=Path), default=Path("configs/train_smoke.yaml"))
@click.option(
    "--mode",
    type=click.Choice(["spectrum", "spectrum_structure"]),
    default="spectrum",
    show_default=True,
)
@click.option("--run-dir", type=click.Path(path_type=Path), default=None)
def train_cmd(paths: Path, dataset_version: str | None, config: Path, mode: str, run_dir: Path | None):
    paths_cfg = load_yaml(paths)
    p = resolve_paths(paths_cfg)
    dv = dataset_version or str(p["dataset_version"])
    train_cfg = merge_train_defaults(load_yaml(config))

    ds_dir = p["processed_root"] / dv
    if not ds_dir.exists():
        raise click.ClickException(f"Нет датасета {ds_dir}; сначала build-dataset")

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    rd = run_dir or Path("runs") / f"run_{mode}_{ts}"
    metrics = train_models(
        dataset_dir=ds_dir,
        run_dir=rd,
        mode=mode,  # type: ignore[arg-type]
        train_cfg=train_cfg,
        random_seed=int(train_cfg["random_seed"]),
        train_frac=float(train_cfg["train_frac"]),
    )
    click.echo(f"Training finished. Run dir: {rd}\nMetrics: {metrics}")


@main.command("evaluate")
@click.option("--paths", type=click.Path(exists=True, path_type=Path), default=Path("configs/paths.local.yaml"))
@click.option("--dataset-version", type=str, default=None)
@click.option("--run-dir", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--mode", type=click.Choice(["spectrum", "spectrum_structure"]), required=True)
def evaluate_cmd(paths: Path, dataset_version: str | None, run_dir: Path, mode: str):
    paths_cfg = load_yaml(paths)
    p = resolve_paths(paths_cfg)
    dv = dataset_version or str(p["dataset_version"])
    ds_dir = p["processed_root"] / dv
    summary = evaluate_run(ds_dir, run_dir, mode=mode)
    click.echo(summary)


@main.command("predict")
@click.option("--paths", type=click.Path(exists=True, path_type=Path), default=Path("configs/paths.local.yaml"))
@click.option("--jcamp", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--run-dir", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output-dir", type=click.Path(path_type=Path), default=Path("reports/predict_out"))
@click.option("--torch", "use_torch", is_flag=True, help="использовать torch_bundle.pt из run-dir вместо sklearn")
def predict_cmd(paths: Path, jcamp: Path, run_dir: Path, output_dir: Path, use_torch: bool):
    paths_cfg = load_yaml(paths)
    p = resolve_paths(paths_cfg)
    tb = run_dir / "torch_bundle.pt" if use_torch else None
    if use_torch and not tb.exists():
        raise click.ClickException(f"Нет {tb}; сначала ir-pipeline torch-train ...")
    predict_file_visualize(
        jcamp_path=jcamp,
        bands_yaml=p["bands_config"],
        bundle_path=run_dir / "models.joblib",
        out_dir=output_dir,
        torch_bundle=tb if use_torch else None,
    )
    click.echo(f"Wrote plot and summary to {output_dir}")


@main.command("torch-train")
@click.option("--paths", type=click.Path(exists=True, path_type=Path), default=Path("configs/paths.local.yaml"))
@click.option("--dataset-version", type=str, default=None)
@click.option("--config", type=click.Path(exists=True, path_type=Path), default=Path("configs/train_smoke.yaml"))
@click.option(
    "--mode",
    type=click.Choice(["spectrum", "spectrum_structure"]),
    default="spectrum",
    show_default=True,
)
@click.option("--run-dir", type=click.Path(path_type=Path), default=None)
@click.option("--device", type=str, default=None, help="cuda | cpu | пусто=авто")
def torch_train_cmd(
    paths: Path,
    dataset_version: str | None,
    config: Path,
    mode: str,
    run_dir: Path | None,
    device: str | None,
):
    """PyTorch 1D CNN: многозадачная регрессия положений полос (cm^-1); график и история в run-dir."""
    if not torch_train_mod.is_torch_available():
        raise click.ClickException("Установите torch: pip install -e '.[torch]'")

    paths_cfg = load_yaml(paths)
    p = resolve_paths(paths_cfg)
    dv = dataset_version or str(p["dataset_version"])
    ds_dir = p["processed_root"] / dv
    if not ds_dir.exists():
        raise click.ClickException(f"Нет датасета {ds_dir}; сначала build-dataset")

    train_cfg = merge_train_defaults(load_yaml(config))
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    rd = run_dir or Path("runs") / f"torch_{mode}_{ts}"

    summary = torch_train_mod.train_torch_run(
        dataset_dir=ds_dir,
        run_dir=rd,
        mode=mode,
        train_cfg=train_cfg,
        device=device,
    )
    click.echo(f"Torch training done → {rd}\n{summary}")


if __name__ == "__main__":
    main()
