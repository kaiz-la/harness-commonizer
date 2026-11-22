"""Part-number (module) lineage: project-2 modules mapped to project-1 modules."""
from __future__ import annotations

from collections import Counter

import pandas as pd


def _top(values, n=3) -> str:
    c = Counter(v for v in values if v and v != "-")
    return "; ".join(f"{k} ({v})" for k, v in c.most_common(n))


def map_modules(wire_map: pd.DataFrame, w1: pd.DataFrame, w2: pd.DataFrame) -> pd.DataFrame:
    """For every project-2 part number:

    * p1 by wires  : majority project-1 part number of its carried-over wires
    * declared     : predecessor_part_number written in the project-2 list
    * p1_part_number: declared predecessor when the wire evidence shows a carry-over, else the
                      wire majority (the declared value is a claim; the wires are evidence)
    """
    w2p = w2[w2.part_number != ""]
    wm = wire_map[wire_map.p2_part_number != ""]
    carried = wm[wm.status == "carry-over"]
    p1_count = w1[w1.part_number != ""].groupby("part_number").size()

    first = w2p.groupby("part_number").agg(
        harness_attribute=("harness_attribute", lambda s: s.mode().iat[0] if len(s) else ""),
        harness_design=("harness_design", lambda s: s.mode().iat[0] if len(s) else ""),
        Vercodung=("vercodung", lambda s: s.mode().iat[0] if len(s) else ""),
        kem=("kem", lambda s: s.mode().iat[0] if len(s) else ""),
        zgs=("zgs", lambda s: s.mode().iat[0] if len(s) else ""),
        p2_wire_count=("wireid", "size"),
        declared_predecessor=("predecessor_pn", lambda s: Counter(x for x in s if x and x != "-").most_common(1)[0][0]
                              if any(x and x != "-" for x in s) else ""),
    )
    cw = carried.groupby("p2_part_number").agg(
        carried_wires=("p1_wireid", "size"),
        p1_by_wires=("p1_part_number", lambda s: Counter(x for x in s if x).most_common(1)[0][0]
                     if any(s) else ""),
        p1_candidates=("p1_part_number", _top),
        _decl_agree=("p2_predecessor_part_number", lambda s: s.tolist()),
    )
    newc = wm[wm.status.str.startswith("new")].groupby("p2_part_number").size().rename("new_wires")
    m = first.join(cw, how="left").join(newc, how="left")
    m["carried_wires"] = m["carried_wires"].fillna(0).astype(int)
    m["new_wires"] = m["new_wires"].fillna(0).astype(int)
    m["p1_by_wires"] = m["p1_by_wires"].fillna("")
    m["p1_candidates"] = m["p1_candidates"].fillna("")
    m["status"] = m["carried_wires"].gt(0).map({True: "carry-over", False: "new"})
    m["p1_part_number"] = m["declared_predecessor"].where(
        (m["declared_predecessor"] != "") & (m["status"] == "carry-over"), m["p1_by_wires"])
    m.loc[m.status == "new", "p1_part_number"] = ""
    m["carry_share"] = (m["carried_wires"] / m["p2_wire_count"]).round(3)
    m["p1_wire_count"] = m["p1_part_number"].map(p1_count).fillna(m["carried_wires"]).astype(int)
    m["lineage_source"] = ""
    m.loc[(m.status == "carry-over") & (m.declared_predecessor != ""), "lineage_source"] = "declared+wires"
    m.loc[(m.status == "carry-over") & (m.declared_predecessor == ""), "lineage_source"] = "wires"
    m.loc[(m.status == "new") & (m.declared_predecessor != ""), "lineage_source"] = "declared-only (no wire evidence)"
    m = m.drop(columns=["_decl_agree"]).reset_index().rename(columns={"part_number": "p2_part_number"})
    cols = ["p2_part_number", "p1_part_number", "status", "harness_attribute", "harness_design", "Vercodung",
            "kem", "zgs", "p1_wire_count", "p2_wire_count", "carried_wires", "new_wires", "carry_share",
            "declared_predecessor", "p1_by_wires", "p1_candidates", "lineage_source"]
    return m[cols].sort_values(["harness_attribute", "p2_part_number"]).reset_index(drop=True)
