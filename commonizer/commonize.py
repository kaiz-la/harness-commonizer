"""Part-number commonization: merge candidates and function clusters.

Each part number (module) becomes a weighted sparse vector made of
  connections   - its wires as translated endpoint pairs (common connector space)
  vercodung     - parsed code-rule tokens (P_/N_ codes), IDF weighted
  harness       - harness_attribute / harness_design tokens
  long_name     - function words of the connected components
  wire profile  - cross-section | colour | wire part number
Nearest neighbours (sparse cosine, chunked) propose pairs; exact set metrics and
Vercodung logic then decide what can be combined. Clusters come from the similarity
graph (connected components) with HDBSCAN splitting oversized components.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as l2norm

from .config import Config
from .normalize import combine_rules, parse_vercodung, rules_conflict
from .wire_match import endpoint_key

_ident = lambda x: x  # noqa: E731


def _vec(docs: List[List[str]], min_df=1):
    if not any(docs):
        return sparse.csr_matrix((len(docs), 0))
    v = TfidfVectorizer(analyzer=_ident, min_df=min_df, sublinear_tf=True, dtype=np.float32)
    try:
        return v.fit_transform(docs)
    except ValueError:  # empty vocabulary
        return sparse.csr_matrix((len(docs), 0))


def _harness_family(attr: str) -> str:
    return (attr.split()[0] if attr else "").split(",")[0]


def _group_mode(w: pd.DataFrame, key: str, col: str) -> pd.Series:
    """Most frequent value of col per key, vectorised (ties -> first in sort order)."""
    c = w.groupby([key, col], sort=True).size().reset_index(name="n")
    c = c.sort_values([key, "n"], ascending=[True, False], kind="stable")
    return c.drop_duplicates(key).set_index(key)[col]


def build_modules(wires: pd.DataFrame, project: str) -> pd.DataFrame:
    """wires must carry e1_conn/e1_pin/e2_conn/e2_pin in common connector space."""
    w = wires.loc[wires.part_number != "", ["part_number", "e1_conn", "e1_pin", "e2_conn", "e2_pin",
                                            "cross_section", "color", "wire_pn", "harness_attribute",
                                            "harness_design", "vercodung", "long_name"]].copy()
    w["ekey"] = endpoint_key(w)
    w["prof"] = w["cross_section"] + "|" + w["color"] + "|" + w["wire_pn"]
    g = w.groupby("part_number", sort=True)
    m = pd.DataFrame(index=pd.Index(sorted(g.groups.keys()), name="part_number"))
    m["project"] = project
    m["connections"] = g["ekey"].agg(frozenset)
    ends = pd.concat([w[["part_number", "e1_conn"]].rename(columns={"e1_conn": "c"}),
                      w[["part_number", "e2_conn"]].rename(columns={"e2_conn": "c"})])
    m["connectors"] = ends[ends.c != ""].groupby("part_number")["c"].agg(frozenset)
    m["connectors"] = m["connectors"].map(lambda x: x if isinstance(x, frozenset) else frozenset())
    m["wire_count"] = g.size()
    for col in ("harness_attribute", "harness_design", "vercodung"):
        m[col] = _group_mode(w, "part_number", col)
    m["vercodung_variants"] = g["vercodung"].nunique()
    m["long_names"] = g["long_name"].agg(lambda s: [x for x in s if x])
    m["profiles"] = g["prof"].agg(list)
    return m.reset_index()


def module_matrix(m: pd.DataFrame, cfg: Config):
    """Returns (X, Xc): X = full weighted feature matrix (rows L2-normalised);
    Xc = sparse 'blocking' matrix of discriminative tokens only, used to propose candidates."""
    conn_docs = [list(c) + [f"C:{x}" for x in k] for c, k in zip(m.connections, m.connectors)]
    rules = [parse_vercodung(v) for v in m.vercodung]
    ver_docs = [r.tokens() for r in rules]
    har_docs = [[f"A:{a}", f"F:{_harness_family(a)}"] + [f"D:{d.strip()}" for d in hd.split(",") if d.strip()]
                for a, hd in zip(m.harness_attribute, m.harness_design)]
    ln_docs = [[t.lower() for n in names for t in re.findall(r"[A-Za-zÄÖÜäöüß]{3,}", n)] for names in m.long_names]
    Bc, Bv = _vec(conn_docs), _vec(ver_docs)
    blocks = [
        (Bc, cfg.w_connections),
        (Bv, cfg.w_vercodung),
        (_vec(har_docs), cfg.w_harness),
        (_vec(ln_docs), cfg.w_long_name),
        (_vec(list(m.profiles)), cfg.w_wire_profile),
    ]
    mats = [l2norm(b) * np.sqrt(wt) for b, wt in blocks if b.shape[1] and wt > 0]
    X = l2norm(sparse.hstack(mats).tocsr().astype(np.float32))

    # blocking: connection/connector/code tokens that occur in at most `cap` modules
    cap = max(cfg.candidate_max_df, 2)
    parts = []
    for B in (Bc, Bv):
        if B.shape[1]:
            B = (B > 0).astype(np.float32).tocsc()
            df = np.diff(B.indptr)
            keep = np.where((df >= 2) & (df <= cap))[0]
            parts.append(B[:, keep])
    Xc = sparse.hstack(parts).tocsr() if parts else sparse.csr_matrix((len(m), 0))
    return X, Xc


def topk_pairs(X: sparse.csr_matrix, k: int, min_sim: float, Xc: Optional[sparse.csr_matrix] = None,
               chunk: int = 4000, groups: Optional[np.ndarray] = None):
    """Sparse cosine kNN. Candidates come from shared blocking tokens (Xc); exact similarity
    is then computed on X for those candidates only. If `groups` is given, only pairs with the
    same group id (>= 0) are kept. Returns arrays i<j, sim."""
    n = X.shape[0]
    if Xc is None:
        Xc = X
    XcT = Xc.T.tocsr()
    I_all, J_all, V_all = [], [], []
    for s in range(0, n, chunk):
        C = (Xc[s:s + chunk] @ XcT).tocoo()
        r, c = C.row + s, C.col
        ok = r != c
        if groups is not None:
            ok &= (groups[r] == groups[c]) & (groups[r] >= 0)
        r, c = r[ok], c[ok]
        if not len(r):
            continue
        sims = np.empty(len(r), dtype=np.float32)
        for b in range(0, len(r), 500_000):
            rr, cc = r[b:b + 500_000], c[b:b + 500_000]
            sims[b:b + 500_000] = np.asarray(X[rr].multiply(X[cc]).sum(axis=1)).ravel()
        keep = sims >= min_sim
        r, c, sims = r[keep], c[keep], sims[keep]
        if not len(r):
            continue
        d = pd.DataFrame({"i": r, "j": c, "v": sims}).sort_values(["i", "v"], ascending=[True, False])
        d = d.groupby("i", sort=False).head(k)
        I_all.append(d.i.values)
        J_all.append(d.j.values)
        V_all.append(d.v.values)
    if not I_all:
        return np.array([], int), np.array([], int), np.array([], float)
    I, J, V = np.concatenate(I_all), np.concatenate(J_all), np.concatenate(V_all)
    lo, hi = np.minimum(I, J), np.maximum(I, J)
    df = pd.DataFrame({"i": lo, "j": hi, "v": V}).drop_duplicates(["i", "j"])
    return df.i.values, df.j.values, df.v.values.astype(float)


def _recommend(r, cfg: Config) -> str:
    if r["vercodung_conflict"] and r["connection_jaccard"] >= cfg.merge_jaccard:
        return "Variant pair (review)"
    if r["vercodung_conflict"]:
        return "Conflict"
    if r["harness_compat"] == 0:
        return "Review (different harness)"
    if r["containment"] >= cfg.absorb_containment and r["connection_jaccard"] < 0.999:
        return "Absorb"
    if r["connection_jaccard"] >= cfg.merge_jaccard:
        return "Merge"
    if r["vercodung_equivalent"] and r["harness_compat"] == 1:
        return "Combine (same code rule)"
    if r["connector_jaccard"] >= cfg.merge_jaccard and r["harness_compat"] == 1:
        return "Combine (shared connectors)"
    return "Review"


def merge_candidates(m: pd.DataFrame, X: sparse.csr_matrix, cfg: Config,
                     exclude_pairs: Optional[set] = None, progress: Optional[Callable] = None,
                     Xc: Optional[sparse.csr_matrix] = None) -> pd.DataFrame:
    log = progress or (lambda *_: None)
    rules = [parse_vercodung(v) for v in m.vercodung]
    groups = None
    if cfg.merge_same_vercodung_only:
        # group = logically identical code rule (term / alternative order does not matter)
        canon = pd.Series([r.canonical() for r in rules])
        groups = pd.factorize(canon)[0]
        if not cfg.merge_include_empty_vercodung:
            groups[(canon == "").values] = -1
        # every same-rule group of a manageable size is also a blocking token, so pairs that
        # share the rule but little else still get compared
        size = np.bincount(groups[groups >= 0]) if (groups >= 0).any() else np.array([], int)
        ok = np.where((size >= 2) & (size <= max(cfg.candidate_max_df, 2)))[0]
        rows = np.where(np.isin(groups, ok))[0]
        if len(rows):
            col = pd.factorize(groups[rows])[0]
            G = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, col)), shape=(len(m), col.max() + 1))
            Xc = G if Xc is None or not Xc.shape[1] else sparse.hstack([Xc, G]).tocsr()
        log(f"  same-Vercodung mode: {int((size >= 2).sum()):,} code-rule groups with ≥2 modules")
    I, J, V = topk_pairs(X, cfg.merge_knn, cfg.merge_min_similarity, Xc, groups=groups)
    log(f"  {len(I):,} candidate module pairs from kNN")
    pn = m.part_number.values
    proj = m.project.values
    out = []
    for i, j, v in zip(I, J, V):
        if exclude_pairs and ((pn[i], pn[j]) in exclude_pairs or (pn[j], pn[i]) in exclude_pairs):
            continue
        A, B = m.connections.iat[i], m.connections.iat[j]
        inter = len(A & B)
        union = len(A | B) or 1
        small = min(len(A), len(B)) or 1
        ra, rb = rules[i], rules[j]
        conflict = rules_conflict(ra, rb)
        pa, pb = ra.positive, rb.positive
        code_j = len(pa & pb) / (len(pa | pb) or 1)
        ha, hb = m.harness_attribute.iat[i], m.harness_attribute.iat[j]
        compat = 1 if ha == hb else (0.5 if _harness_family(ha) == _harness_family(hb) else 0)
        # larger module first
        if m.wire_count.iat[j] > m.wire_count.iat[i]:
            i, j = j, i
            A, B = B, A
        out.append({
            "part_number_a": pn[i], "project_a": proj[i], "part_number_b": pn[j], "project_b": proj[j],
            "similarity": round(v, 4),
            "connection_jaccard": round(inter / union, 4),
            "containment": round(inter / small, 4),
            "shared_connections": inter,
            "only_in_a": len(A - B), "only_in_b": len(B - A),
            "connector_jaccard": round(len(m.connectors.iat[i] & m.connectors.iat[j])
                                       / (len(m.connectors.iat[i] | m.connectors.iat[j]) or 1), 4),
            "harness_attribute_a": m.harness_attribute.iat[i], "harness_attribute_b": m.harness_attribute.iat[j],
            "harness_compat": compat,
            "vercodung_a": m.vercodung.iat[i], "vercodung_b": m.vercodung.iat[j],
            "vercodung_code_jaccard": round(code_j, 4),
            "vercodung_equivalent": ra.canonical() == rb.canonical() and bool(ra.terms),
            "vercodung_conflict": ",".join(sorted(conflict)),
            "suggested_vercodung": combine_rules([rules[i], rules[j]]),
            "wires_a": int(m.wire_count.iat[i]), "wires_b": int(m.wire_count.iat[j]),
            "example_shared": "; ".join(sorted(A & B)[:3]),
        })
    df = pd.DataFrame(out)
    if df.empty:
        return df
    df["recommendation"] = df.apply(lambda r: _recommend(r, cfg), axis=1)
    rank = {"Absorb": 0, "Merge": 1, "Combine (same code rule)": 2, "Combine (shared connectors)": 3,
            "Variant pair (review)": 4, "Review": 5, "Review (different harness)": 6, "Conflict": 7}
    df["_r"] = df.recommendation.map(rank)
    df["scope"] = np.where(df.project_a == df.project_b, "within " + df.project_a, "cross-project")
    df = df.sort_values(["_r", "similarity"], ascending=[True, False]).drop(columns="_r")
    lead = ["recommendation", "scope"]
    return df[lead + [c for c in df.columns if c not in lead]].reset_index(drop=True)


def function_clusters(m: pd.DataFrame, X: sparse.csr_matrix, pairs: pd.DataFrame, cfg: Config,
                      progress: Optional[Callable] = None):
    """Returns (clusters summary, membership)."""
    log = progress or (lambda *_: None)
    n = len(m)
    pos = pd.Series(np.arange(n), index=m.part_number.values + "@" + m.project.values)
    if pairs.empty:
        return pd.DataFrame(), pd.DataFrame()
    e = pairs[(pairs.similarity >= cfg.cluster_similarity)
              & (pairs.vercodung_conflict == "")
              & (pairs.harness_compat > 0)]
    ia = pos.loc[(e.part_number_a + "@" + e.project_a).values].values
    ib = pos.loc[(e.part_number_b + "@" + e.project_b).values].values
    G = sparse.coo_matrix((e.similarity.values, (ia, ib)), shape=(n, n)).tocsr()
    G = G.maximum(G.T)
    ncomp, labels = connected_components(G, directed=False)
    labels = labels.copy()
    sizes = np.bincount(labels)
    next_label = labels.max() + 1
    big = np.where(sizes > cfg.cluster_max_component)[0]
    if len(big):
        from sklearn.cluster import HDBSCAN
        from sklearn.decomposition import TruncatedSVD
        for b in big:
            idx = np.where(labels == b)[0]
            k = int(min(cfg.svd_components, X.shape[1] - 1, len(idx) - 1))
            Z = TruncatedSVD(n_components=max(k, 2), random_state=0).fit_transform(X[idx]) if k >= 2 else X[idx].toarray()
            Z = l2norm(Z)
            sub = HDBSCAN(min_cluster_size=2, min_samples=1).fit_predict(Z)
            for s in np.unique(sub):
                sel = idx[sub == s]
                if s == -1:  # noise -> singletons
                    for x in sel:
                        labels[x] = next_label
                        next_label += 1
                else:
                    labels[sel] = next_label
                    next_label += 1
        log(f"  split {len(big)} oversized components with HDBSCAN")
    sizes = np.bincount(labels)
    clusters, members = [], []
    cid = 0
    for lab in np.where(sizes >= 2)[0]:
        idx = np.where(labels == lab)[0]
        cid += 1
        sub = G[idx][:, idx]
        strength = np.asarray(sub.sum(axis=1)).ravel() + 1e-6 * m.wire_count.values[idx]
        rep = idx[int(np.argmax(strength))]
        cnt = Counter(c for x in idx for c in m.connections.iat[x])
        core = [c for c, k in cnt.items() if k >= max(2, 0.5 * len(idx))]
        allc = set(cnt)
        rules = [parse_vercodung(m.vercodung.iat[x]) for x in idx]
        clusters.append({
            "cluster_id": f"FC{cid:05d}",
            "size": len(idx),
            "projects": ", ".join(sorted(set(m.project.values[idx]))),
            "representative_part": m.part_number.iat[rep],
            "representative_project": m.project.iat[rep],
            "harness_attributes": ", ".join(sorted(set(m.harness_attribute.values[idx]))),
            "total_wires": int(m.wire_count.values[idx].sum()),
            "unique_connections": len(allc),
            "core_connections": len(core),
            "commonality": round(len(core) / (len(allc) or 1), 3),
            "possible_reduction": len(idx) - 1,
            "suggested_vercodung": combine_rules(rules),
            "members": "; ".join(sorted(f"{m.part_number.iat[x]} [{m.project.iat[x]}]" for x in idx)),
            "example_core": "; ".join(sorted(core)[:3]),
        })
        for x in idx:
            members.append({"cluster_id": f"FC{cid:05d}", "part_number": m.part_number.iat[x],
                            "project": m.project.iat[x], "is_representative": x == rep,
                            "wire_count": int(m.wire_count.iat[x]),
                            "harness_attribute": m.harness_attribute.iat[x], "vercodung": m.vercodung.iat[x],
                            "share_of_core": round(len(set(core) & m.connections.iat[x]) / (len(core) or 1), 3)})
    cl = pd.DataFrame(clusters)
    if len(cl):
        cl = cl.sort_values(["size", "commonality"], ascending=False).reset_index(drop=True)
    log(f"  {len(cl):,} function clusters (≥2 parts)")
    return cl, pd.DataFrame(members)


def orphan_suggestions(wires: pd.DataFrame, m: pd.DataFrame, X: sparse.csr_matrix, project: str,
                       cfg: Config, limit: int = 50_000) -> pd.DataFrame:
    """Wires without a part number: best fitting existing module (same harness, shared connectors, codes)."""
    o = wires[wires.part_number == ""].head(limit)
    if o.empty or m.empty:
        return pd.DataFrame()
    mm = m[m.project == project].reset_index(drop=True)
    if mm.empty:
        return pd.DataFrame()
    by_conn: Dict[str, List[int]] = {}
    for i, cs in enumerate(mm.connectors):
        for c in cs:
            by_conn.setdefault(c, []).append(i)
    rows = []
    m_rules = [parse_vercodung(v) for v in mm.vercodung]
    m_conn = list(mm.connectors)
    m_attr = list(mm.harness_attribute)
    for r in o.itertuples():
        cands = set(by_conn.get(r.e1_conn, [])) | set(by_conn.get(r.e2_conn, []))
        best, bs = None, 0.0
        rr = parse_vercodung(r.vercodung)
        for i in cands:
            s = 0.0
            cs = m_conn[i]
            s += 0.4 * ((r.e1_conn in cs) + (r.e2_conn in cs)) / 2
            s += 0.3 * (m_attr[i] == r.harness_attribute)
            mr = m_rules[i]
            s += 0.3 * (len(rr.positive & mr.positive) / (len(rr.positive | mr.positive) or 1))
            if rules_conflict(rr, mr):
                s -= 0.5
            if s > bs:
                best, bs = i, s
        rows.append({"wireid": r.wireid, "Von/Bis": r.von_bis, "harness_attribute": r.harness_attribute,
                     "vercodung": r.vercodung,
                     "suggested_part_number": mm.part_number.iat[best] if best is not None else "",
                     "score": round(bs, 3)})
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def part_number_report(m: pd.DataFrame, lineage: pd.DataFrame, lab1: str, lab2: str,
                       pairs: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    """One row per part number per project: attributes, lineage, cluster and best merge partner."""
    r = m[["project", "part_number", "wire_count", "harness_attribute", "harness_design", "vercodung",
           "vercodung_variants"]].copy()
    r["connectors"] = m.connectors.map(len).values
    r["connections"] = m.connections.map(len).values
    key = r.project + "@" + r.part_number
    # lineage between the two projects
    lin = lineage[lineage.p1_part_number.astype(str) != ""]
    succ = lin.groupby("p1_part_number").p2_part_number.agg(lambda s: "; ".join(sorted(set(s))))
    pred = lineage.set_index("p2_part_number")
    is1, is2 = r.project == lab1, r.project == lab2
    r["lineage_status"] = ""
    r["linked_part_numbers"] = ""
    r.loc[is2, "lineage_status"] = r.loc[is2, "part_number"].map(pred.status).fillna("")
    r.loc[is2, "linked_part_numbers"] = r.loc[is2, "part_number"].map(pred.p1_part_number).fillna("")
    s1 = r.loc[is1, "part_number"].map(succ)
    r.loc[is1, "lineage_status"] = np.where(s1.notna(), "carried into " + lab2, "not carried (dropped)")
    r.loc[is1, "linked_part_numbers"] = s1.fillna("")
    # clusters
    r["cluster_id"], r["cluster_size"], r["is_representative"] = "", 0, False
    if len(members):
        mk = members.project + "@" + members.part_number
        sz = members.cluster_id.map(members.cluster_id.value_counts())
        r["cluster_id"] = key.map(pd.Series(members.cluster_id.values, mk.values)).fillna("")
        r["cluster_size"] = key.map(pd.Series(sz.values, mk.values)).fillna(0).astype(int)
        r["is_representative"] = key.isin(set(mk[members.is_representative.astype(bool)].values))
    # merge candidates (pairs are sorted best-first)
    r["merge_candidates"], r["best_merge_partner"], r["best_recommendation"], r["best_similarity"] = 0, "", "", np.nan
    if len(pairs):
        a = pd.DataFrame({"k": pairs.project_a + "@" + pairs.part_number_a,
                          "other": pairs.part_number_b + " [" + pairs.project_b + "]",
                          "rec": pairs.recommendation, "sim": pairs.similarity})
        b = pd.DataFrame({"k": pairs.project_b + "@" + pairs.part_number_b,
                          "other": pairs.part_number_a + " [" + pairs.project_a + "]",
                          "rec": pairs.recommendation, "sim": pairs.similarity})
        e = pd.concat([a, b], ignore_index=True).reset_index().sort_values("index", kind="stable")
        first = e.drop_duplicates("k").set_index("k")
        r["merge_candidates"] = key.map(e.k.value_counts()).fillna(0).astype(int)
        r["best_merge_partner"] = key.map(first.other).fillna("")
        r["best_recommendation"] = key.map(first.rec).fillna("")
        r["best_similarity"] = key.map(first.sim)
    return r.sort_values(["project", "part_number"]).reset_index(drop=True)
