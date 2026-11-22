"""Scores a result against an answer key (same sheet layout as Project1_vs_Project2_answer_key.xlsx)."""
from __future__ import annotations

from typing import Dict

import pandas as pd


def _prf(pred: set, truth: set):
    tp = len(pred & truth)
    p = tp / len(pred) if pred else 1.0
    r = tp / len(truth) if truth else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return tp, p, r, f


def evaluate(result: Dict[str, pd.DataFrame], key: Dict[str, pd.DataFrame]):
    rows, mism = [], []

    def add(area, metric, value, detail=""):
        rows.append({"area": area, "metric": metric, "value": value, "detail": detail})

    if "VPC_Mapping" in key:
        k = key["VPC_Mapping"]
        truth = set(zip(k.project1_connector, k.project2_connector))
        v = result["VPC_Mapping"]
        pred = set(zip(v.project1_connector, v.project2_connector))
        tp, p, r, f = _prf(pred, truth)
        add("VPC renames", "precision", round(p, 4), f"{tp}/{len(pred)}")
        add("VPC renames", "recall", round(r, 4), f"{tp}/{len(truth)}")
        add("VPC renames", "F1", round(f, 4))
        for a, b in sorted(pred - truth)[:500]:
            mism.append({"area": "VPC", "item": a, "predicted": b, "expected": dict(truth).get(a, "")})
        for a, b in sorted(truth - pred)[:500]:
            mism.append({"area": "VPC", "item": a, "predicted": dict(pred).get(a, ""), "expected": b})

    if "Wire_Mapping" in key:
        k = key["Wire_Mapping"]
        kc = k[k.status == "carry-over"]
        truth = set(zip(kc.p2_wireid, kc.p1_wireid))
        w = result["Wire_Mapping"]
        wc = w[w.status == "carry-over"]
        pred = set(zip(wc.p2_wireid, wc.p1_wireid))
        tp, p, r, f = _prf(pred, truth)
        add("Wire carry-over pairs", "precision", round(p, 4), f"{tp}/{len(pred)}")
        add("Wire carry-over pairs", "recall", round(r, 4), f"{tp}/{len(truth)}")
        add("Wire carry-over pairs", "F1", round(f, 4))
        st = w.set_index("p2_wireid")["status"]
        ks = k.set_index("p2_wireid")["status"]
        both = ks.index.intersection(st.index)
        acc = (st.loc[both] == ks.loc[both]).mean() if len(both) else 0
        add("Wire status", "accuracy (carry-over/new)", round(float(acc), 4), f"{len(both)} wires")
        tv = k.set_index("p2_wireid")["touches_VPC"]
        rv = w.set_index("p2_wireid")["touches_VPC"]
        bc = k.p2_wireid[k.p2_wireid.isin(rv.index)]
        add("Wire touches_VPC", "accuracy", round(float((rv.loc[bc] == tv.loc[bc]).mean()), 4), f"{len(bc)} wires")
        wrong = w.merge(k[["p2_wireid", "p1_wireid", "status"]], on="p2_wireid", suffixes=("", "_expected"))
        wrong = wrong[(wrong.p1_wireid != wrong.p1_wireid_expected) | (wrong.status != wrong.status_expected)]
        for r_ in wrong.head(1000).itertuples():
            mism.append({"area": "Wire", "item": r_.p2_wireid, "predicted": f"{r_.status}:{r_.p1_wireid}",
                         "expected": f"{r_.status_expected}:{r_.p1_wireid_expected}"})

    if "Part_Number_Modules" in key:
        k = key["Part_Number_Modules"]
        m = result["Part_Number_Modules"].set_index("p2_part_number")
        k = k[k.p2_part_number.isin(m.index)]
        acc_status = (m.loc[k.p2_part_number, "status"].values == k.status.values).mean()
        kc = k[k.status == "carry-over"]
        acc_map = (m.loc[kc.p2_part_number, "p1_part_number"].values == kc.p1_part_number.values).mean()
        add("Modules", "status accuracy", round(float(acc_status), 4), f"{len(k)} modules")
        add("Modules", "P1→P2 mapping accuracy (carry-over)", round(float(acc_map), 4), f"{len(kc)} modules")
        bad = kc[m.loc[kc.p2_part_number, "p1_part_number"].values != kc.p1_part_number.values]
        for r_ in bad.head(500).itertuples():
            mism.append({"area": "Module", "item": r_.p2_part_number,
                         "predicted": m.loc[r_.p2_part_number, "p1_part_number"], "expected": r_.p1_part_number})

    if "Summary" in key:
        ks = dict(zip(key["Summary"].iloc[:, 0].str.strip(), key["Summary"].iloc[:, 1]))
        ours = dict(zip(result["Summary"]["Item"].str.strip(), result["Summary"]["Count"]))
        for item, exp in ks.items():
            if item in ours:
                add("Summary counts", item, ours[item], f"expected {exp}")
    return pd.DataFrame(rows), pd.DataFrame(mism, columns=["area", "item", "predicted", "expected"])
