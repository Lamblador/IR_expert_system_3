from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem, rdBase

UA = "ir-pipeline/0.1 (research)"
LAMBLADOR_IRSPECTRA_URL = "https://raw.githubusercontent.com/Lamblador/IR_expert_system_2/main/expanded_df.pkl"
LAMBLADOR_SEED_CACHE_NAME = "lamblador_irspectra_structures.parquet"


def structure_cache_path(processed_root: Path, dataset_version: str) -> Path:
    return processed_root / dataset_version / "structure_cache.parquet"


def load_structure_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    out: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        key = str(row["lookup_key"])
        out[key] = {
            "smiles": row.get("smiles"),
            "inchi": row.get("inchi"),
            "inchikey": row.get("inchikey"),
            "source": row.get("source"),
            "error": row.get("error"),
        }
    return out


def save_structure_cache(path: Path, cache: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"lookup_key": k, **v} for k, v in cache.items()]
    pd.DataFrame(rows).to_parquet(path, index=False)


def seed_structure_cache_from_lamblador(
    cache: dict[str, dict[str, Any]],
    processed_root: Path,
    source_url: str = LAMBLADOR_IRSPECTRA_URL,
) -> int:
    """Заполняет CAS -> SMILES/InChI из Lamblador/IRSpectra до сетевых запросов PubChem."""
    seed_path = processed_root / LAMBLADOR_SEED_CACHE_NAME
    df = _load_lamblador_seed_table(seed_path, source_url)
    added = 0
    for _, row in df.iterrows():
        cas = _normalize_cas(row.get("cas"))
        if not cas:
            continue
        key = f"cas:{cas}"
        current = cache.get(key)
        if current and current.get("smiles"):
            continue
        cache[key] = {
            "smiles": row.get("smiles"),
            "inchi": row.get("inchi"),
            "inchikey": row.get("inchikey"),
            "source": "lamblador_irspectra",
            "error": None,
        }
        added += 1
    return added


def _load_lamblador_seed_table(seed_path: Path, source_url: str) -> pd.DataFrame:
    if seed_path.exists():
        return pd.read_parquet(seed_path)

    df = pd.read_pickle(source_url)
    columns = {str(c).lower(): c for c in df.columns}
    required = {"cas": columns.get("cas"), "smiles": columns.get("smiles"), "inchi": columns.get("inchi")}
    if not all(required.values()):
        missing = ", ".join(k for k, v in required.items() if v is None)
        raise RuntimeError(f"Lamblador/IRSpectra seed не содержит колонки: {missing}")

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    rdBase.DisableLog("rdApp.*")
    try:
        for _, row in df.iterrows():
            cas = _normalize_cas(row[required["cas"]])
            if not cas or cas in seen:
                continue
            res = _resolution_from_identifiers(row[required["smiles"]], row[required["inchi"]], "lamblador_irspectra")
            if res.get("smiles"):
                rows.append({"cas": cas, **res})
                seen.add(cas)
    finally:
        rdBase.EnableLog("rdApp.*")

    out = pd.DataFrame(rows)
    seed_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(seed_path, index=False)
    return out


def _normalize_cas(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    cas = str(value).strip()
    if not cas or not re.fullmatch(r"\d{2,10}-\d{2}-\d", cas):
        return None
    return cas


def _first_text(*values: Any) -> str | None:
    for value in values:
        if value is None or pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _resolution_from_identifiers(smiles: Any, inchi: Any, source: str) -> dict[str, Any]:
    sm = _first_text(smiles)
    inch = _first_text(inchi)
    mol = Chem.MolFromSmiles(sm) if sm else None
    if mol is None and inch:
        mol = Chem.MolFromInchi(inch)
    if mol is None:
        return {"smiles": None, "inchi": inch, "inchikey": None, "source": source, "error": "rdkit_parse_failed"}
    return {
        "smiles": Chem.MolToSmiles(mol),
        "inchi": inch or Chem.MolToInchi(mol),
        "inchikey": Chem.MolToInchiKey(mol),
        "source": source,
        "error": None,
    }


def _pubchem_rest_cas(cas: str, retries: int = 4) -> dict[str, Any]:
    cas_enc = urllib.parse.quote(cas.strip(), safe="")
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/xref/RN/"
        f"{cas_enc}/property/IsomericSMILES,CanonicalSMILES,InChI,InChIKey/JSON"
    )
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            props = data["PropertyTable"]["Properties"][0]
            smiles = props.get("IsomericSMILES") or props.get("CanonicalSMILES") or props.get("SMILES")
            inchi = props.get("InChI")
            inchikey = props.get("InChIKey")
            if not smiles:
                return {
                    "smiles": None,
                    "inchi": inchi,
                    "inchikey": inchikey,
                    "source": "pubchem_rest",
                    "error": "no_smiles",
                }
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {
                    "smiles": None,
                    "inchi": inchi,
                    "inchikey": inchikey,
                    "source": "pubchem_rest",
                    "error": "rdkit_parse_failed",
                }
            return {
                "smiles": Chem.MolToSmiles(mol),
                "inchi": inchi or Chem.MolToInchi(mol),
                "inchikey": inchikey or Chem.MolToInchiKey(mol),
                "source": "pubchem_rest",
                "error": None,
            }
        except urllib.error.HTTPError as e:
            last_err = f"HTTPError:{e.code}"
            if 400 <= e.code < 500:
                break
            time.sleep(1.2 * (attempt + 1))
        except Exception as e:
            last_err = str(e)
            time.sleep(1.2 * (attempt + 1))
    return {"smiles": None, "inchi": None, "inchikey": None, "source": "pubchem_rest", "error": last_err}


def _pubchempy_fallback(cas: str) -> dict[str, Any]:
    try:
        import pubchempy as pcp  # type: ignore
    except Exception:
        return {"smiles": None, "inchi": None, "inchikey": None, "source": "pubchempy", "error": "no_pubchempy"}
    try:
        comps = pcp.get_compounds(cas, "name")
        if not comps:
            return {"smiles": None, "inchi": None, "inchikey": None, "source": "pubchempy", "error": "not_found"}
        c = comps[0]
        smiles = c.isomeric_smiles or c.canonical_smiles
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None:
            return {"smiles": None, "inchi": None, "inchikey": None, "source": "pubchempy", "error": "bad_smiles"}
        return {
            "smiles": Chem.MolToSmiles(mol),
            "inchi": Chem.MolToInchi(mol),
            "inchikey": Chem.MolToInchiKey(mol),
            "source": "pubchempy",
            "error": None,
        }
    except Exception as e:
        return {"smiles": None, "inchi": None, "inchikey": None, "source": "pubchempy", "error": str(e)}


def resolve_from_pubchem_cas(cas: str, sleep_s: float = 0.12) -> dict[str, Any]:
    """CAS Registry Number → SMILES/InChI (PubChem PUG REST, затем fallback pubchempy)."""
    cas = cas.strip()
    time.sleep(max(0.0, sleep_s))
    r = _pubchem_rest_cas(cas)
    if r.get("smiles"):
        return r
    fb = _pubchempy_fallback(cas)
    if fb.get("smiles"):
        return fb
    return r if r.get("error") else fb


def _pubchem_rest_by_compound_name(name: str, retries: int = 4) -> dict[str, Any]:
    """Поиск по названию соединения (PubChem `compound/name/...`)."""
    name_enc = urllib.parse.quote(name.strip()[:220], safe="")
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{name_enc}/property/IsomericSMILES,CanonicalSMILES,InChI,InChIKey/JSON"
    )
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            props = data["PropertyTable"]["Properties"][0]
            smiles = props.get("IsomericSMILES") or props.get("CanonicalSMILES")
            inchi = props.get("InChI")
            inchikey = props.get("InChIKey")
            if not smiles:
                return {
                    "smiles": None,
                    "inchi": inchi,
                    "inchikey": inchikey,
                    "source": "pubchem_name",
                    "error": "no_smiles",
                }
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {
                    "smiles": None,
                    "inchi": inchi,
                    "inchikey": inchikey,
                    "source": "pubchem_name",
                    "error": "rdkit_parse_failed",
                }
            return {
                "smiles": Chem.MolToSmiles(mol),
                "inchi": inchi or Chem.MolToInchi(mol),
                "inchikey": inchikey or Chem.MolToInchiKey(mol),
                "source": "pubchem_name",
                "error": None,
            }
        except urllib.error.HTTPError as e:
            last_err = f"HTTPError:{e.code}"
            time.sleep(1.2 * (attempt + 1))
        except Exception as e:
            last_err = str(e)
            time.sleep(1.2 * (attempt + 1))
    return {"smiles": None, "inchi": None, "inchikey": None, "source": "pubchem_name", "error": last_err}


def resolve_from_pubchem_name(title: str, sleep_s: float = 0.12) -> dict[str, Any]:
    """Название (TITLE из JCAMP) → SMILES/InChI."""
    title = title.strip()
    if len(title) < 4:
        return {"smiles": None, "inchi": None, "inchikey": None, "source": "pubchem_name", "error": "title_too_short"}
    time.sleep(max(0.0, sleep_s))
    return _pubchem_rest_by_compound_name(title)


def resolve_structure_for_record(
    cas: str | None,
    title: str | None,
    cache: dict[str, dict[str, Any]],
    sleep_s: float,
) -> dict[str, Any]:
    """CAS → при неудаче TITLE (кэш с ключами `cas:` и `name:`)."""
    cas = (cas or "").strip()
    tit = (title or "").strip()

    if cas:
        ck = f"cas:{cas}"
        if ck not in cache:
            cache[ck] = resolve_from_pubchem_cas(cas, sleep_s=sleep_s)
        if cache[ck].get("smiles"):
            return cache[ck]

    if tit and len(tit) >= 4:
        nk = f"name:{tit[:220]}"
        if nk not in cache:
            cache[nk] = resolve_from_pubchem_name(tit, sleep_s=sleep_s)
        if cache[nk].get("smiles"):
            return cache[nk]

    if cas:
        return cache[f"cas:{cas}"]
    if tit and len(tit) >= 4:
        return cache[f"name:{tit[:220]}"]
    return {"smiles": None, "inchi": None, "inchikey": None, "source": None, "error": "no_cas_or_title"}


def mol_from_resolution(res: dict[str, Any]) -> Chem.Mol | None:
    sm = res.get("smiles")
    if not sm:
        return None
    return Chem.MolFromSmiles(sm)


def resolve_or_cache(lookup_key: str, cache: dict[str, dict[str, Any]], resolver_fn) -> dict[str, Any]:
    if lookup_key in cache:
        return cache[lookup_key]
    res = resolver_fn(lookup_key)
    cache[lookup_key] = res
    return res
