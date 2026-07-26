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
from tqdm import tqdm

UA = "ir-pipeline/0.1 (research)"
LAMBLADOR_IRSPECTRA_URL = "https://raw.githubusercontent.com/Lamblador/IR_expert_system_2/main/expanded_df.pkl"
LAMBLADOR_SEED_CACHE_NAME = "lamblador_irspectra_structures.parquet"
LOCAL_TITLE_SMILES: dict[str, str] = {
    "ndcl3 6h2o": "O.O.O.O.O.O.[Cl-].[Cl-].[Cl-].[Nd+3]",
    "ndcl3 xh2o": "[Cl-].[Cl-].[Cl-].[Nd+3]",
    "nd2(co3)3": "[O-]C([O-])=O.[O-]C([O-])=O.[O-]C([O-])=O.[Nd+3].[Nd+3]",
    "nd sulfuricum": "[O-]S(=O)(=O)[O-].[O-]S(=O)(=O)[O-].[O-]S(=O)(=O)[O-].[Nd+3].[Nd+3]",
    "pr(tfacet)3 nh2o": "CC(=O)C([O-])=C(C)C.CC(=O)C([O-])=C(C)C.CC(=O)C([O-])=C(C)C.[Pr+3]",
    "gd(no3)3 8h2o": "O.O.O.O.O.O.O.O.[O-][N+](=O)[O-].[O-][N+](=O)[O-].[O-][N+](=O)[O-].[Gd+3]",
    "dy(no3)3 4h2o": "O.O.O.O.[O-][N+](=O)[O-].[O-][N+](=O)[O-].[O-][N+](=O)[O-].[Dy+3]",
    "y2(co3)3 3h2o": "O.O.O.[O-]C([O-])=O.[O-]C([O-])=O.[O-]C([O-])=O.[Y+3].[Y+3]",
    "y(no3)3 6h2o": "O.O.O.O.O.O.[O-][N+](=O)[O-].[O-][N+](=O)[O-].[O-][N+](=O)[O-].[Y+3]",
    "ycl3 6h2o": "O.O.O.O.O.O.[Cl-].[Cl-].[Cl-].[Y+3]",
    "lu(no3)3 4h2o": "O.O.O.O.[O-][N+](=O)[O-].[O-][N+](=O)[O-].[O-][N+](=O)[O-].[Lu+3]",
    "lucl3 6h2o": "O.O.O.O.O.O.[Cl-].[Cl-].[Cl-].[Lu+3]",
    "lucl3 xh2o": "[Cl-].[Cl-].[Cl-].[Lu+3]",
    "smcl3": "[Cl-].[Cl-].[Cl-].[Sm+3]",
    "sm(no3)3 nh2o": "[O-][N+](=O)[O-].[O-][N+](=O)[O-].[O-][N+](=O)[O-].[Sm+3]",
    "y(thd)3 h2o": "CC(=O)C([O-])=C(C(C)C)C(C)C.CC(=O)C([O-])=C(C(C)C)C(C)C.CC(=O)C([O-])=C(C(C)C)C(C)C.[Y+3]",
    "gd2(co3)3 nh2o": "[O-]C([O-])=O.[O-]C([O-])=O.[O-]C([O-])=O.[Gd+3].[Gd+3]",
    "ercl3 6h2o": "O.O.O.O.O.O.[Cl-].[Cl-].[Cl-].[Er+3]",
    "la(ac)3": "CC(=O)[O-].CC(=O)[O-].CC(=O)[O-].[La+3]",
    "la2(co3)3": "[O-]C([O-])=O.[O-]C([O-])=O.[O-]C([O-])=O.[La+3].[La+3]",
    "tb2(so4)3 8h2o": "O.O.O.O.O.O.O.O.[O-]S(=O)(=O)[O-].[O-]S(=O)(=O)[O-].[O-]S(=O)(=O)[O-].[Tb+3].[Tb+3]",
    "tb2(co3)3 3h2o": "O.O.O.[O-]C([O-])=O.[O-]C([O-])=O.[O-]C([O-])=O.[Tb+3].[Tb+3]",
    "tb(naf)3 nh2o": "FC(C([O-])=O)(F)C(F)(F)F.FC(C([O-])=O)(F)C(F)(F)F.FC(C([O-])=O)(F)C(F)(F)F.[Tb+3]",
    "pr2(co3)3": "[O-]C([O-])=O.[O-]C([O-])=O.[O-]C([O-])=O.[Pr+3].[Pr+3]",
    "pracet3 1h2o": "O.CC(=O)[O-].CC(=O)[O-].CC(=O)[O-].[Pr+3]",
    "sm(no3)3 6h2o": "O.O.O.O.O.O.[O-][N+](=O)[O-].[O-][N+](=O)[O-].[O-][N+](=O)[O-].[Sm+3]",
    "smacet3 6.5h2o": "CC(=O)[O-].CC(=O)[O-].CC(=O)[O-].[Sm+3]",
    "nd(no3)3 6h2o": "O.O.O.O.O.O.[O-][N+](=O)[O-].[O-][N+](=O)[O-].[O-][N+](=O)[O-].[Nd+3]",
    "nd2(so4)3 8h2o": "O.O.O.O.O.O.O.O.[O-]S(=O)(=O)[O-].[O-]S(=O)(=O)[O-].[O-]S(=O)(=O)[O-].[Nd+3].[Nd+3]",
}


def structure_cache_path(processed_root: Path, dataset_version: str) -> Path:
    return processed_root / dataset_version / "structure_cache.parquet"


def load_structure_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    out: dict[str, dict[str, Any]] = {}
    records = df.to_dict("records")
    for row in tqdm(records, total=len(records), desc="Load structure cache", unit="row"):
        key = str(row["lookup_key"])
        out[key] = {
            "smiles": _safe_text(row.get("smiles")),
            "inchi": _safe_text(row.get("inchi")),
            "inchikey": _safe_text(row.get("inchikey")),
            "source": _safe_text(row.get("source")),
            "error": _safe_text(row.get("error")),
        }
    return out


def save_structure_cache(path: Path, cache: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"lookup_key": k, **v} for k, v in tqdm(cache.items(), total=len(cache), desc="Prepare structure cache", unit="row")]
    pd.DataFrame(rows).to_parquet(path, index=False)


def seed_structure_cache_from_lamblador(
    cache: dict[str, dict[str, Any]],
    processed_root: Path,
    source_url: str = LAMBLADOR_IRSPECTRA_URL,
) -> int:
    """Заполняет CAS/name -> SMILES/InChI из Lamblador/IRSpectra до сетевых запросов PubChem."""
    seed_path = processed_root / LAMBLADOR_SEED_CACHE_NAME
    df = _load_lamblador_seed_table(seed_path, source_url)
    existing_lamblador = sum(
        1 for row in cache.values() if row.get("source") == "lamblador_irspectra" and row.get("smiles")
    )
    if existing_lamblador >= len(df):
        tqdm.write(f"[ir-pipeline] Lamblador seed already present: {existing_lamblador} cache keys")
        return 0

    added = 0
    for row in tqdm(df.to_dict("records"), total=len(df), desc="Seed structure cache", unit="row"):
        resolution = {
            "smiles": row.get("smiles"),
            "inchi": row.get("inchi"),
            "inchikey": row.get("inchikey"),
            "source": "lamblador_irspectra",
            "error": None,
        }
        keys = []
        cas = _normalize_cas(row.get("cas"))
        if cas:
            keys.append(f"cas:{cas}")
        ik = _normalize_inchikey(row.get("inchikey"))
        if ik:
            keys.append(f"inchikey:{ik}")
        for name_col in ("name", "title"):
            name_key = _name_lookup_key(row.get(name_col))
            if name_key:
                keys.append(name_key)
        for key in dict.fromkeys(keys):
            current = cache.get(key)
            if current and current.get("smiles"):
                continue
            cache[key] = dict(resolution)
            added += 1
    return added


def _load_lamblador_seed_table(seed_path: Path, source_url: str) -> pd.DataFrame:
    if seed_path.exists():
        cached = pd.read_parquet(seed_path)
        expected_columns = {"cas", "name", "title", "smiles", "inchi", "inchikey"}
        if expected_columns.issubset(cached.columns):
            tqdm.write(f"[ir-pipeline] Lamblador seed cache loaded: {seed_path} ({len(cached)} rows)")
            return cached

    tqdm.write(f"[ir-pipeline] Loading Lamblador/IRSpectra seed from {source_url}")
    df = pd.read_pickle(source_url)
    columns = {str(c).lower(): c for c in df.columns}
    required = {"cas": columns.get("cas"), "smiles": columns.get("smiles"), "inchi": columns.get("inchi")}
    optional = {"name": columns.get("name"), "title": columns.get("title"), "formula": columns.get("formula")}
    if not all(required.values()):
        missing = ", ".join(k for k, v in required.items() if v is None)
        raise RuntimeError(f"Lamblador/IRSpectra seed не содержит колонки: {missing}")

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    rdBase.DisableLog("rdApp.*")
    try:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Build Lamblador seed", unit="row"):
            cas = _normalize_cas(row[required["cas"]])
            if not cas or cas in seen:
                continue
            res = _resolution_from_identifiers(row[required["smiles"]], row[required["inchi"]], "lamblador_irspectra")
            if res.get("smiles"):
                rows.append(
                    {
                        "cas": cas,
                        "name": _first_text(row[optional["name"]]) if optional["name"] else None,
                        "title": _first_text(row[optional["title"]]) if optional["title"] else None,
                        "formula": _first_text(row[optional["formula"]]) if optional["formula"] else None,
                        **res,
                    }
                )
                seen.add(cas)
    finally:
        rdBase.EnableLog("rdApp.*")

    out = pd.DataFrame(rows)
    seed_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(seed_path, index=False)
    tqdm.write(f"[ir-pipeline] Lamblador seed cache written: {seed_path} ({len(out)} rows)")
    return out


def _normalize_cas(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    cas = str(value).strip()
    if not cas or not re.fullmatch(r"\d{2,10}-\d{2}-\d", cas):
        return None
    return cas


def _normalize_inchikey(value: Any) -> str | None:
    text = _first_text(value)
    if not text:
        return None
    ik = text.strip().upper()
    # стандартный InChIKey: 14-1-8 символов через дефис
    if not re.fullmatch(r"[A-Z]{14}-[A-Z]{10}-[A-Z]", ik):
        # допускаем укороченные/нестандартные ключи из экспорта, если похожи
        if len(ik) < 14 or " " in ik:
            return None
    return ik


def _cache_put_resolution(cache: dict[str, dict[str, Any]], key: str, res: dict[str, Any]) -> None:
    current = cache.get(key)
    if current and current.get("smiles") and not res.get("smiles"):
        return
    cache[key] = dict(res)


def _index_resolution_keys(
    cache: dict[str, dict[str, Any]],
    res: dict[str, Any],
    *,
    cas: str | None = None,
    inchikey: str | None = None,
) -> None:
    ik = _normalize_inchikey(inchikey or res.get("inchikey"))
    if ik:
        _cache_put_resolution(cache, f"inchikey:{ik}", res)
    cas_n = _normalize_cas(cas)
    if cas_n and res.get("smiles"):
        _cache_put_resolution(cache, f"cas:{cas_n}", res)


def _first_text(*values: Any) -> str | None:
    for value in values:
        if value is None or pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _safe_text(value: Any) -> str | None:
    """
    Нормализует значение из parquet/cache в строку или None.
    Защищает от float/NaN и других неожиданных типов.
    """
    return _first_text(value)


def _name_lookup_key(value: Any) -> str | None:
    name = _first_text(value)
    if not name or len(name) < 4:
        return None
    return f"name:{name[:220].casefold()}"


def _normalize_title_for_local_lookup(value: str) -> str:
    t = value.casefold().replace("_", " ")
    t = re.sub(r"^\d+\s*", "", t)
    t = t.replace("h2o)", "h2o")
    t = re.sub(r"[^a-z0-9()+. ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


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


def _pubchem_rest_by_inchikey(inchikey: str, retries: int = 4) -> dict[str, Any]:
    ik = _normalize_inchikey(inchikey)
    if not ik:
        return {"smiles": None, "inchi": None, "inchikey": None, "source": "pubchem_inchikey", "error": "bad_inchikey"}
    ik_enc = urllib.parse.quote(ik, safe="")
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/"
        f"{ik_enc}/property/IsomericSMILES,CanonicalSMILES,InChI,InChIKey/JSON"
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
            inchikey_out = props.get("InChIKey") or ik
            if not smiles:
                return {
                    "smiles": None,
                    "inchi": inchi,
                    "inchikey": inchikey_out,
                    "source": "pubchem_inchikey",
                    "error": "no_smiles",
                }
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {
                    "smiles": None,
                    "inchi": inchi,
                    "inchikey": inchikey_out,
                    "source": "pubchem_inchikey",
                    "error": "rdkit_parse_failed",
                }
            return {
                "smiles": Chem.MolToSmiles(mol),
                "inchi": inchi or Chem.MolToInchi(mol),
                "inchikey": inchikey_out or Chem.MolToInchiKey(mol),
                "source": "pubchem_inchikey",
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
    return {"smiles": None, "inchi": None, "inchikey": ik, "source": "pubchem_inchikey", "error": last_err}


def resolve_from_pubchem_inchikey(inchikey: str, sleep_s: float = 0.12) -> dict[str, Any]:
    """InChIKey → SMILES/InChI через PubChem."""
    time.sleep(max(0.0, sleep_s))
    return _pubchem_rest_by_inchikey(inchikey)


def resolve_structure_for_record(
    cas: str | None,
    title: str | None,
    cache: dict[str, dict[str, Any]],
    sleep_s: float,
    allow_network: bool = False,
    *,
    inchi: str | None = None,
    inchikey: str | None = None,
) -> dict[str, Any]:
    """
    Структура для записи спектра.

    Приоритет: InChI из JCAMP (офлайн RDKit) → кэш InChIKey → CAS → TITLE →
    (опционально сеть) PubChem по InChIKey/CAS/name.
    """
    cas = (cas or "").strip()
    tit = (title or "").strip()
    inchi_s = _first_text(inchi)
    ik = _normalize_inchikey(inchikey)

    # 1) Полный InChI из JCAMP → SMILES без сети
    if inchi_s:
        jcamp_res = _resolution_from_identifiers(None, inchi_s, "jcamp_inchi")
        if jcamp_res.get("smiles"):
            if ik and not jcamp_res.get("inchikey"):
                jcamp_res["inchikey"] = ik
            _index_resolution_keys(cache, jcamp_res, cas=cas or None, inchikey=ik)
            return jcamp_res

    # 2) Кэш / сеть по InChIKey
    if ik:
        ik_key = f"inchikey:{ik}"
        if ik_key in cache and cache[ik_key].get("smiles"):
            return cache[ik_key]
        if allow_network:
            net = resolve_from_pubchem_inchikey(ik, sleep_s=sleep_s)
            _cache_put_resolution(cache, ik_key, net)
            if net.get("smiles"):
                _index_resolution_keys(cache, net, cas=cas or None, inchikey=ik)
                return net

    # 3) CAS
    if cas:
        ck = f"cas:{cas}"
        if ck not in cache:
            if allow_network:
                cache[ck] = resolve_from_pubchem_cas(cas, sleep_s=sleep_s)
        if ck in cache and cache[ck].get("smiles"):
            res = cache[ck]
            _index_resolution_keys(cache, res, cas=cas, inchikey=ik)
            return res

    # 4) TITLE / локальная карта
    if tit and len(tit) >= 4:
        local_key = _normalize_title_for_local_lookup(tit)
        local_smiles = LOCAL_TITLE_SMILES.get(local_key)
        if local_smiles:
            local_res = _resolution_from_identifiers(local_smiles, None, "local_title_map")
            if local_res.get("smiles"):
                nk = _name_lookup_key(tit)
                if nk:
                    cache[nk] = local_res
                _index_resolution_keys(cache, local_res, cas=cas or None, inchikey=ik)
                return local_res
        nk = _name_lookup_key(tit)
        if nk is None:
            return {
                "smiles": None,
                "inchi": inchi_s,
                "inchikey": ik,
                "source": None,
                "error": "title_too_short",
            }
        if nk not in cache:
            if allow_network:
                cache[nk] = resolve_from_pubchem_name(tit, sleep_s=sleep_s)
        if nk in cache and cache[nk].get("smiles"):
            res = cache[nk]
            _index_resolution_keys(cache, res, cas=cas or None, inchikey=ik)
            return res

    # Частичный ответ: идентификаторы из JCAMP без SMILES
    if cas:
        ck = f"cas:{cas}"
        if ck in cache:
            partial = dict(cache[ck])
            partial.setdefault("inchi", inchi_s)
            partial.setdefault("inchikey", ik)
            return partial
    if tit and len(tit) >= 4:
        nk = _name_lookup_key(tit)
        if nk and nk in cache:
            partial = dict(cache[nk])
            partial.setdefault("inchi", inchi_s)
            partial.setdefault("inchikey", ik)
            return partial

    if ik and f"inchikey:{ik}" in cache:
        return cache[f"inchikey:{ik}"]

    err = "not_in_fast_structure_cache" if not allow_network else "no_structure_id"
    return {"smiles": None, "inchi": inchi_s, "inchikey": ik, "source": None, "error": err}


def mol_from_resolution(res: dict[str, Any]) -> Chem.Mol | None:
    sm = _safe_text(res.get("smiles"))
    if sm:
        with rdBase.BlockLogs():
            mol = Chem.MolFromSmiles(sm)
            if mol is not None:
                return mol
    inch = _safe_text(res.get("inchi"))
    if inch:
        with rdBase.BlockLogs():
            return Chem.MolFromInchi(inch)
    return None


def resolve_or_cache(lookup_key: str, cache: dict[str, dict[str, Any]], resolver_fn) -> dict[str, Any]:
    if lookup_key in cache:
        return cache[lookup_key]
    res = resolver_fn(lookup_key)
    cache[lookup_key] = res
    return res
