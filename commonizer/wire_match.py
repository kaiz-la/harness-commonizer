"""Wire-level carry-over / new / dropped classification after connector alignment."""
from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment

from .config import Config
from .normalize import add_endpoint_columns, parse_connector


def translate(w1: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """Project-1 wires expressed in project-2 connector names (order-insensitive endpoints)."""
    t = w1.copy()
    t["connector_a"] = t["connector_a"].map(lambda c: mapping.get(c, c))
    t["connector_b"] = t["connector_b"].map(lambda c: mapping.get(c, c))
    return add_endpoint_columns(t)


def endpoint_key(w: pd.DataFrame) -> pd.Series:
    return w["e1_conn"] + "/" + w["e1_pin"] + " ~ " + w["e2_conn"] + "/" + w["e2_pin"]


def _attr_similarity(a: pd.Series, b: pd.Series, cols) -> float:
    score, total = 0.0, 0.0
    for c in cols:
        wgt = 2.0 if c in ("wire_pn", "cross_section", "color") else 1.0
        total += wgt
        x, y = a[c], b[c]
        if x == y:
            score += wgt
        elif c in ("long_name", "remark", "cable_name"):
            score += wgt * fuzz.token_set_ratio(x, y) / 100.0
    return score / total if total else 0.0


def match_wires(w1: pd.DataFrame, w2: pd.DataFrame, mapping: Dict[str, str], cfg: Config,
                progress: Optional[Callable] = None) -> pd.DataFrame:
    log = progress or (lambda *_: None)
    t1 = translate(w1, mapping)
    t1["ekey"] = endpoint_key(t1)
    w2 = w2.copy()
    w2["ekey"] = endpoint_key(w2)
    t1["xkey"] = t1["ekey"] + "|" + t1["fp"]
    w2["xkey"] = w2["ekey"] + "|" + w2["fp"]
    # parallel identical wires: pair them by occurrence number
    t1["k"] = t1["xkey"] + "#" + t1.groupby("xkey").cumcount().astype(str)
    w2["k"] = w2["xkey"] + "#" + w2.groupby("xkey").cumcount().astype(str)

    exact = w2[["wireid", "k"]].merge(t1[["wireid", "k"]], on="k", suffixes=("_p2", "_p1"))
    pairs = [(r.wireid_p2, r.wireid_p1, 1.0, "exact", "") for r in exact.itertuples()]
    log(f"  exact wire matches: {len(pairs):,}")

    # fuzzy fallback: same translated endpoints, attributes partly changed
    used1, used2 = set(exact.wireid_p1), set(exact.wireid_p2)
    r1 = t1[~t1.wireid.isin(used1)].set_index("wireid")
    r2 = w2[~w2.wireid.isin(used2)].set_index("wireid")
    common = set(r1["ekey"]) & set(r2["ekey"])
    cols = [c for c in cfg.fingerprint_cols if c in r1.columns]
    g1 = r1[r1.ekey.isin(common)].groupby("ekey")
    g2 = {k: v for k, v in r2[r2.ekey.isin(common)].groupby("ekey")}
    n_fuzzy = 0
    for ek, a in g1:
        b = g2[ek]
        S = np.array([[_attr_similarity(a.loc[i], b.loc[j], cols) for j in b.index] for i in a.index])
        ri, ci = linear_sum_assignment(-S)
        for i, j in zip(ri, ci):
            if S[i, j] >= cfg.fuzzy_wire_min_score:
                x, y = a.iloc[i], b.iloc[j]
                changed = ",".join(c for c in cols if x[c] != y[c])
                pairs.append((b.index[j], a.index[i], float(S[i, j]), "endpoints+fuzzy", changed))
                n_fuzzy += 1
    log(f"  fuzzy wire matches: {n_fuzzy:,}")

    pm = pd.DataFrame(pairs, columns=["p2_wireid", "p1_wireid", "confidence", "match_method", "changed_attributes"])
    renamed = {k for k, v in mapping.items() if k != v}
    a2 = w2.set_index("wireid")
    a1 = w1.set_index("wireid")

    out = w2[["wireid"]].rename(columns={"wireid": "p2_wireid"}).merge(pm, on="p2_wireid", how="left")
    out["status"] = np.where(out["p1_wireid"].notna(), "carry-over", "new")
    out["p1_wireid"] = out["p1_wireid"].fillna("")
    out["confidence"] = out["confidence"].fillna(0.0).round(3)
    out["match_method"] = out["match_method"].fillna("")
    out["changed_attributes"] = out["changed_attributes"].fillna("")
    p1v = a1.reindex(out["p1_wireid"])
    out["p1_Von/Bis"] = p1v["von_bis"].fillna("").values
    out["p2_Von/Bis"] = a2.loc[out["p2_wireid"], "von_bis"].values
    # VPC = variant position connector (name carries a .position suffix, e.g. A101X9.4)
    def is_vpc(series):
        return series.map(lambda c: bool(c) and parse_connector(c)[2] != "" and not parse_connector(c)[0].startswith("#")).values
    p2v = a2.loc[out["p2_wireid"]]
    touches = (is_vpc(p2v["connector_a"]) | is_vpc(p2v["connector_b"])
               | is_vpc(p1v["connector_a"].fillna("")) | is_vpc(p1v["connector_b"].fillna("")))
    out["touches_VPC"] = np.where(touches, "yes", "no")
    out["p1_connector_renamed"] = np.where(
        p1v["connector_a"].isin(renamed).values | p1v["connector_b"].isin(renamed).values, "yes", "no")
    out["p1_part_number"] = p1v["part_number"].fillna("").values
    out["p2_part_number"] = a2.loc[out["p2_wireid"], "part_number"].values
    out["p2_predecessor_part_number"] = a2.loc[out["p2_wireid"], "predecessor_pn"].values
    out["harness_attribute"] = a2.loc[out["p2_wireid"], "harness_attribute"].values
    cols_out = ["p2_wireid", "p1_wireid", "status", "touches_VPC", "p1_Von/Bis", "p2_Von/Bis",
                "p1_part_number", "p2_part_number", "p2_predecessor_part_number", "harness_attribute",
                "p1_connector_renamed",
                "confidence", "match_method", "changed_attributes"]
    out = out[cols_out]

    dropped = w1[~w1.wireid.isin(set(pm.p1_wireid))][["wireid", "von_bis", "part_number", "harness_attribute"]]
    dropped = dropped.rename(columns={"wireid": "p1_wireid", "von_bis": "p1_Von/Bis", "part_number": "p1_part_number"})
    return out, dropped.reset_index(drop=True)
