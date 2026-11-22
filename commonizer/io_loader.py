"""Robust loading of DBX logical wirelists (xlsx / csv) and answer keys.

* Detects the wirelist sheet by its header, not its name.
* Maps column name variants onto canonical names.
* Streams with openpyxl read_only, caches a parquet copy keyed by file hash.
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd

CACHE_DIR = Path(os.environ.get("HC_CACHE_DIR", Path.home() / ".cache" / "harness_commonizer"))

# canonical name -> accepted header variants (compared after normalising)
ALIASES: Dict[str, List[str]] = {
    "wireid": ["wireid", "wire id", "wire_id", "leitungsid", "leitung id"],
    "logic_design": ["logic_design", "logic design", "logikplan"],
    "component_a": ["component a", "component_a", "bauteil a"],
    "connector_a": ["connector a", "connector_a", "stecker a", "von stecker"],
    "pin_a": ["pin a", "pin_a", "kammer a", "von pin"],
    "component_b": ["component b", "component_b", "bauteil b"],
    "connector_b": ["connector b", "connector_b", "stecker b", "bis stecker"],
    "pin_b": ["pin b", "pin_b", "kammer b", "bis pin"],
    "von_bis": ["von/bis", "von bis", "vonbis", "from/to", "from to"],
    "cross_section": ["querschnitt", "cross section", "cross_section", "csa"],
    "color": ["leitungsfarbe", "farbe", "color", "colour", "wire color"],
    "wire_pn": ["leitungssachnummer", "wire part number", "wire_pn"],
    "cable_pn": ["kabelsachnummer", "cable part number", "cable_pn"],
    "cable_name": ["kabelname", "cable name", "cable_name"],
    "print": ["bedruckung", "print", "marking"],
    "harness_attribute": ["harness_attribute", "harness attribute", "kabelbaumattribut"],
    "vercodung": ["vercodung", "verkodung", "code rule", "coderegel", "variant code"],
    "remark": ["bemerkung", "remark", "comment"],
    "kem": ["kem"],
    "part_number": ["part_number", "part number", "sachnummer", "modul", "module"],
    "predecessor_pn": ["predecessor_part_number", "predecessor part number", "vorgaenger", "vorgänger"],
    "state": ["state", "status"],
    "zgs": ["zgs"],
    "harness_design": ["harness_design", "harness design"],
    "long_name": ["long_name", "long name", "langname", "benennung"],
}
REQUIRED_ANY = [{"wireid"}, {"von_bis", "connector_a"}]


def _norm(h) -> str:
    return re.sub(r"\s+", " ", str(h or "")).strip().lower().replace("-", " ")


_LOOKUP = {}
for canon, variants in ALIASES.items():
    for v in variants:
        _LOOKUP[_norm(v)] = canon


def canonical_columns(cols) -> Dict[str, str]:
    out, used = {}, set()
    for c in cols:
        canon = _LOOKUP.get(_norm(c))
        if canon and canon not in used:
            out[c] = canon
            used.add(canon)
    return out


@dataclass
class Wirelist:
    path: str
    name: str
    wires: pd.DataFrame
    meta: Dict[str, str] = field(default_factory=dict)
    sheets: Dict[str, pd.DataFrame] = field(default_factory=dict)  # raw extra sheets (Bauteile, Belegung)

    @property
    def label(self) -> str:
        return self.meta.get("project") or self.meta.get("kem") or self.name


def _file_hash(path: str) -> str:
    h = hashlib.sha1()
    st = os.stat(path)
    h.update(f"{os.path.abspath(path)}|{st.st_size}|{st.st_mtime_ns}|v2".encode())
    return h.hexdigest()[:16]


def _read_xlsx_sheets(path: str, progress: Optional[Callable] = None) -> Dict[str, List[list]]:
    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    out = {}
    try:
        for ws in wb.worksheets:
            rows = []
            for i, r in enumerate(ws.iter_rows(values_only=True)):
                rows.append(["" if v is None else str(v) for v in r])
                if progress and i and i % 20000 == 0:
                    progress(f"  {Path(path).name} / {ws.title}: {i:,} rows read")
            out[ws.title] = rows
    finally:
        wb.close()
    return out


def _rows_to_df(rows: List[list]) -> pd.DataFrame:
    # header = first row with >= 2 non-empty cells
    hi = 0
    for i, r in enumerate(rows[:20]):
        if sum(1 for v in r if v) >= 2:
            hi = i
            break
    header = [h if h else f"col{j}" for j, h in enumerate(rows[hi])] if rows else []
    body = [r + [""] * (len(header) - len(r)) for r in rows[hi + 1:]]
    body = [r[: len(header)] for r in body if any(r)]
    return pd.DataFrame(body, columns=header, dtype="string")


def _parse_von_bis(s: pd.Series):
    # "X339/2 - A538aX20.4/43"  ->  X339, 2, A538aX20.4, 43
    parts = s.fillna("").str.extract(r"^\s*(?P<ca>[^/]+?)/(?P<pa>[^ ]*)\s*-\s*(?P<cb>[^/]+?)/(?P<pb>\S*)\s*$")
    return parts


def standardise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=canonical_columns(df.columns))
    df = df.loc[:, ~df.columns.duplicated()]
    for c in ALIASES:
        if c not in df.columns:
            df[c] = ""
    df = df.fillna("")
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()
    # derive connectors / pins from Von/Bis when the explicit columns are missing
    need = (df["connector_a"] == "") | (df["connector_b"] == "")
    if need.any() and (df["von_bis"] != "").any():
        p = _parse_von_bis(df.loc[need, "von_bis"]).fillna("")
        for src, dst in [("ca", "connector_a"), ("pa", "pin_a"), ("cb", "connector_b"), ("pb", "pin_b")]:
            df.loc[need & (df[dst] == ""), dst] = p[src]
    # cross-section as normalised text (0.50 == 0.5)
    cs = pd.to_numeric(df["cross_section"].str.replace(",", "."), errors="coerce")
    df["cross_section"] = cs.map(lambda v: "" if pd.isna(v) else f"{v:g}").where(cs.notna(), df["cross_section"])
    if (df["wireid"] == "").any():
        blank = df["wireid"] == ""
        df.loc[blank, "wireid"] = [f"_row{i}" for i in df.index[blank]]
    # duplicate wire ids get a suffix so they stay addressable
    dup = df["wireid"].duplicated(keep=False)
    if dup.any():
        df.loc[dup, "wireid"] = df.loc[dup, "wireid"] + "#" + df[dup].groupby("wireid").cumcount().astype(str)
    return df.reset_index(drop=True)


def _is_wirelist(df: pd.DataFrame) -> bool:
    canon = set(canonical_columns(df.columns).values())
    return all(canon & req for req in REQUIRED_ANY)


def load_wirelist(path: str, progress: Optional[Callable] = None, use_cache: bool = True) -> Wirelist:
    path = str(path)
    name = Path(path).stem
    key = _file_hash(path)
    cache = CACHE_DIR / f"{key}.parquet"
    meta_cache = CACHE_DIR / f"{key}.meta.parquet"
    if use_cache and cache.exists() and meta_cache.exists():
        wires = pd.read_parquet(cache)
        meta = pd.read_parquet(meta_cache).set_index("k")["v"].to_dict()
        if progress:
            progress(f"Loaded {name} from cache ({len(wires):,} wires)")
        return Wirelist(path, name, wires, meta)

    if progress:
        progress(f"Reading {name} …")
    sheets: Dict[str, pd.DataFrame] = {}
    if path.lower().endswith((".csv", ".txt")):
        sheets["wires"] = pd.read_csv(path, dtype=str, sep=None, engine="python").fillna("")
    else:
        for title, rows in _read_xlsx_sheets(path, progress).items():
            if rows:
                sheets[title] = _rows_to_df(rows)
    wl_sheet = next((t for t, d in sheets.items() if _is_wirelist(d)), None)
    if wl_sheet is None:
        raise ValueError(f"{name}: no sheet with a wire list header (needs wireid + Von/Bis or connector columns)")
    wires = standardise(sheets[wl_sheet])

    meta = {}
    for t, d in sheets.items():
        if t == wl_sheet:
            continue
        if d.shape[1] == 2 and len(d) < 50:  # key/value sheet (Stammdaten)
            kv = [list(d.columns)] + d.values.tolist()
            for k, v in kv:
                meta[str(k).strip().lower()] = str(v).strip()
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        wires.to_parquet(cache, index=False)
        pd.DataFrame({"k": list(meta), "v": list(meta.values())}).to_parquet(meta_cache, index=False)
    if progress:
        progress(f"Read {name}: {len(wires):,} wires from sheet '{wl_sheet}'")
    return Wirelist(path, name, wires, meta)


def load_answer_key(path: str) -> Dict[str, pd.DataFrame]:
    out = {}
    for title, rows in _read_xlsx_sheets(str(path)).items():
        if rows:
            out[title] = _rows_to_df(rows).fillna("").astype(str)
    return out
