"""Learns connector (VPC) renames between two projects.

Each connector is described by a sparse TF-IDF profile of the wires plugged into it
(pin, stable wire fingerprint, far-end pin and - once known - the far-end connector in
the common name space). Candidates are compared only inside blocks that renames keep
stable (class letter + position suffix); one-to-one pairs come from the Hungarian
algorithm. Several passes propagate the mapping through the wiring graph (graph
matching), then a global pass catches renames that left their block.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize as l2norm

from .config import Config
from .normalize import connector_block, parse_connector


def _ends(w: pd.DataFrame) -> pd.DataFrame:
    a = pd.DataFrame({"conn": w["e1_conn"].values, "pin": w["e1_pin"].values, "fp": w["fp"].values,
                      "oconn": w["e2_conn"].values, "opin": w["e2_pin"].values})
    b = pd.DataFrame({"conn": w["e2_conn"].values, "pin": w["e2_pin"].values, "fp": w["fp"].values,
                      "oconn": w["e1_conn"].values, "opin": w["e1_pin"].values})
    e = pd.concat([a, b], ignore_index=True)
    return e[e["conn"] != ""]


def _tokens(ends: pd.DataFrame, label: Dict[str, str]) -> pd.DataFrame:
    base = ends["pin"] + "|" + ends["fp"] + "|" + ends["opin"]
    olab = ends["oconn"].map(label)
    t1 = pd.DataFrame({"conn": ends["conn"], "tok": "a:" + base})
    t2 = pd.DataFrame({"conn": ends["conn"], "tok": "b:" + base + "|" + olab}).loc[olab.notna()]
    # far-end-connector-only token: survives attribute edits on the wire
    t3 = pd.DataFrame({"conn": ends["conn"], "tok": "c:" + ends["pin"] + "|" + olab}).loc[olab.notna()]
    return pd.concat([t1, t2, t2, t3], ignore_index=True)  # t2 counted twice = higher weight


def _profiles(tok1, tok2, conns1, conns2):
    vocab = pd.Index(pd.unique(pd.concat([tok1["tok"], tok2["tok"]])))
    i1 = pd.Index(conns1)
    i2 = pd.Index(conns2)

    def mat(tok, idx):
        r = idx.get_indexer(tok["conn"])
        c = vocab.get_indexer(tok["tok"])
        ok = (r >= 0) & (c >= 0)
        return sparse.csr_matrix((np.ones(ok.sum()), (r[ok], c[ok])), shape=(len(idx), len(vocab)))

    m1, m2 = mat(tok1, i1), mat(tok2, i2)
    tf = TfidfTransformer(sublinear_tf=True).fit(sparse.vstack([m1, m2]))
    return l2norm(tf.transform(m1)), l2norm(tf.transform(m2))


def _assign(S: np.ndarray, min_sim: float):
    """Hungarian on a dense similarity block; returns (i, j, sim, margin)."""
    r, c = linear_sum_assignment(-S)
    out = []
    for i, j in zip(r, c):
        s = S[i, j]
        if s < min_sim:
            continue
        row = np.sort(S[i])[::-1]
        col = np.sort(S[:, j])[::-1]
        second = max(row[1] if len(row) > 1 else 0.0, col[1] if len(col) > 1 else 0.0)
        out.append((i, j, float(s), float(s - second)))
    return out


def _greedy(S: sparse.csr_matrix, min_sim: float, k: int):
    S = S.tocoo()
    order = np.argsort(-S.data)
    used_r, used_c, out = set(), set(), []
    best_r = defaultdict(list)
    for i, j, v in zip(S.row, S.col, S.data):
        best_r[i].append(v)
    for idx in order:
        i, j, v = S.row[idx], S.col[idx], S.data[idx]
        if v < min_sim:
            break
        if i in used_r or j in used_c:
            continue
        used_r.add(i)
        used_c.add(j)
        vals = sorted(best_r[i], reverse=True)
        out.append((i, j, float(v), float(v - (vals[1] if len(vals) > 1 else 0.0))))
    return out


def align_connectors(w1: pd.DataFrame, w2: pd.DataFrame, cfg: Config,
                     progress: Optional[Callable] = None) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """w1/w2 need e1_conn,e1_pin,e2_conn,e2_pin,fp. Returns (mapping table, p1->p2 dict incl. identities)."""
    log = progress or (lambda *_: None)
    ends1, ends2 = _ends(w1), _ends(w2)
    conns1 = pd.Index(sorted(ends1["conn"].unique()))
    conns2 = pd.Index(sorted(ends2["conn"].unique()))
    names2 = set(conns2)
    nw1 = ends1.groupby("conn").size()
    nw2 = ends2.groupby("conn").size()
    blocks1 = pd.Series([connector_block(c) for c in conns1], index=conns1)
    blocks2 = pd.Series([connector_block(c) for c in conns2], index=conns2)

    identity = {c: c for c in conns1 if c in names2}
    mapping: Dict[str, str] = {}   # full p1 -> p2 assignment (identities included)
    info: Dict[str, tuple] = {}
    bonus = cfg.align_identity_bonus

    for it in range(cfg.align_iterations):
        lab1 = mapping if mapping else identity
        lab2 = {c: c for c in conns2}
        X1, X2 = _profiles(_tokens(ends1, lab1), _tokens(ends2, lab2), conns1, conns2)

        new_map, new_info = {}, {}
        # ---- block-wise assignment; keeping the same name gets a prior bonus, but a
        #      clearly better-matching renamed connector can still win
        for blk, members1 in blocks1.groupby(blocks1):
            members2 = blocks2.index[blocks2 == blk]
            if len(members2) == 0:
                continue
            r = conns1.get_indexer(members1.index)
            c = conns2.get_indexer(members2)
            S = (X1[r] @ X2[c].T).tocsr()
            same = np.array(members1.index)[:, None] == np.array(members2)[None, :]
            if max(len(r), len(c)) <= cfg.align_dense_block_limit:
                D = S.toarray() + bonus * same
                pairs = _assign(D, cfg.align_min_similarity)
                method = "block-hungarian"
            else:
                ii, jj = np.nonzero(same)
                S = (S + sparse.csr_matrix((np.full(len(ii), bonus), (ii, jj)), shape=S.shape)).tocsr()
                pairs = _greedy(S, cfg.align_min_similarity, cfg.align_knn)
                method = "block-greedy"
            for i, j, s, m in pairs:
                p1, p2 = members1.index[i], members2[j]
                if p1 == p2:
                    s = max(0.0, s - bonus)
                new_map[p1] = p2
                new_info[p1] = (s, m, "identity" if p1 == p2 else method, blk)

        # unmatched same-name connectors with no evidence either way stay identical
        for c in identity:
            if c not in new_map and c not in set(new_map.values()):
                new_map[c] = c
                new_info[c] = (0.0, 0.0, "identity", connector_block(c))

        # ---- global pass for leftovers (renames that changed class letter / position)
        left1 = [c for c in conns1 if c not in new_map]
        taken2 = set(new_map.values())
        left2 = [c for c in conns2 if c not in taken2]
        if left1 and left2:
            r = conns1.get_indexer(left1)
            c = conns2.get_indexer(left2)
            S = (X1[r] @ X2[c].T).tocsr()
            for i, j, s, m in _greedy(S, max(0.35, cfg.align_min_similarity * 2), cfg.align_knn):
                new_map[left1[i]] = left2[j]
                new_info[left1[i]] = (s, m, "global", "*")

        changed = sum(1 for k, v in new_map.items() if mapping.get(k) != v)
        mapping, info = new_map, new_info
        n_ren = sum(1 for k, v in mapping.items() if k != v)
        log(f"  alignment pass {it + 1}: {n_ren} renamed connectors ({changed} assignments changed)")
        if it > 0 and changed == 0:
            break

    rows = []
    for p1, p2 in mapping.items():
        if p1 == p2:
            continue
        s, m, method, blk = info[p1]
        rows.append({
            "project1_connector": p1, "project2_connector": p2,
            "base_p1": parse_connector(p1)[1], "base_p2": parse_connector(p2)[1],
            "similarity": round(s, 4), "margin": round(m, 4), "method": method, "block": blk,
            "p1_wire_ends": int(nw1.get(p1, 0)), "p2_wire_ends": int(nw2.get(p2, 0)),
        })
    table = pd.DataFrame(rows, columns=["project1_connector", "project2_connector", "base_p1", "base_p2",
                                        "similarity", "margin", "method", "block", "p1_wire_ends", "p2_wire_ends"])
    table = table.sort_values("project1_connector").reset_index(drop=True)
    if len(table):
        # family consistency: share of positions of the same P1 base that went to the majority P2 base
        fam = table.groupby("base_p1")["base_p2"].agg(lambda s: s.value_counts(normalize=True).iloc[0])
        table["family_consistency"] = table["base_p1"].map(fam).round(3)
        table["confidence"] = (0.6 * table["similarity"] + 0.25 * table["margin"].clip(0, 1)
                               + 0.15 * table["family_consistency"]).clip(0, 1).round(3)
    return table, dict(mapping)
