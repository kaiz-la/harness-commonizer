#!/usr/bin/env python
"""Builds N× bigger synthetic project pairs + a matching answer key for stress tests.

Each clone k gets its own connector numbering (class letter and .position kept), its own wire
ids and part numbers, and freshly randomised attributes (long_name, colour, cross-section)
applied identically to a carried-over wire on both sides — so clones are not trivially
identical and alignment has to work in blocks k times larger.

  python tools/scale_synth.py P1.xlsx P2.xlsx KEY.xlsx --factor 10 --out-dir bulk/
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xlsxwriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from commonizer.io_loader import ALIASES, load_answer_key, load_wirelist  # noqa: E402
from commonizer.normalize import CONN_RE  # noqa: E402

OUT_COLS = {"wireid": "wireid", "logic_design": "logic_design", "component_a": "component A",
            "connector_a": "connector A", "pin_a": "pin A", "component_b": "component B",
            "connector_b": "connector B", "pin_b": "pin B", "von_bis": "Von/Bis", "cross_section": "Querschnitt",
            "color": "Leitungsfarbe", "wire_pn": "Leitungssachnummer", "cable_pn": "Kabelsachnummer",
            "cable_name": "Kabelname", "print": "Bedruckung", "harness_attribute": "harness_attribute",
            "vercodung": "Vercodung", "remark": "Bemerkung", "kem": "kem", "part_number": "part_number",
            "predecessor_pn": "predecessor_part_number", "state": "state", "zgs": "zgs",
            "harness_design": "harness_design", "long_name": "long_name"}


def shift_conn(name: str, k: int) -> str:
    if k == 0 or not name:
        return name
    m = CONN_RE.match(name)
    if not m:
        return name  # reference points (XR-…, B1-30-Za) stay shared across clones
    num = int(m.group("num")) + 1000 * k
    rest = name[m.end("num"):]
    return f"{m.group('letter')}{num}{rest}"


def shift_pn(pn: str, k: int) -> str:
    return pn if (k == 0 or pn in ("", "-")) else f"{pn}{k:02d}"


def recode(v: str, rng) -> str:
    out = []
    for ch in v:
        if ch.isupper() and ch not in "Z":
            out.append(chr(65 + rng.integers(0, 21)))
        else:
            out.append(ch)
    return "".join(out)


def write_wirelist(path, df, meta):
    wb = xlsxwriter.Workbook(path, {"constant_memory": True})
    ws = wb.add_worksheet("Stammdaten")
    for i, (k, v) in enumerate(meta.items()):
        ws.write_row(i, 0, [k, v])
    ws = wb.add_worksheet("ConnectorWirelist_out_Rev0100_Z")
    cols = list(OUT_COLS.values())
    ws.write_row(0, 0, cols)
    for i, row in enumerate(df[list(OUT_COLS)].itertuples(index=False, name=None), start=1):
        ws.write_row(i, 0, row)
    wb.close()


def write_key(path, sheets):
    wb = xlsxwriter.Workbook(path, {"constant_memory": True})
    for name, df in sheets.items():
        ws = wb.add_worksheet(name)
        ws.write_row(0, 0, list(df.columns))
        for i, row in enumerate(df.itertuples(index=False, name=None), start=1):
            ws.write_row(i, 0, row)
    wb.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("p1")
    ap.add_argument("p2")
    ap.add_argument("key")
    ap.add_argument("--factor", type=int, default=10)
    ap.add_argument("--out-dir", default="bulk")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    w1 = load_wirelist(a.p1).wires
    w2 = load_wirelist(a.p2).wires
    key = load_answer_key(a.key)
    wm = key["Wire_Mapping"]
    pair_of_p1 = dict(zip(wm.p1_wireid, wm.p2_wireid))
    names = sorted(set(w1.long_name) | set(w2.long_name))
    colors = sorted(set(w1.color) | set(w2.color))
    sections = sorted(set(w1.cross_section) | set(w2.cross_section))

    P1, P2, V, W, M, Dr = [], [], [], [], [], []
    for k in range(a.factor):
        # random attribute draw per P2 wire; carried P1 wire copies its partner's draw
        n2 = len(w2)
        draw = pd.DataFrame({
            "long_name": rng.choice(names, n2), "color": rng.choice(colors, n2),
            "cross_section": rng.choice(sections, n2)}, index=w2.wireid.values)
        c2 = w2.copy()
        c1 = w1.copy()
        if k:
            for col in draw.columns:
                c2[col] = draw.loc[c2.wireid, col].values
                partner = c1.wireid.map(pair_of_p1).fillna("")
                carried = partner != ""
                fresh = pd.Series(rng.choice(draw[col].values, len(c1)), index=c1.index)
                c1[col] = np.where(carried, draw[col].reindex(partner).values, fresh.values)
            c2["vercodung"] = [recode(v, rng) for v in c2.vercodung]
            c1["vercodung"] = [recode(v, rng) for v in c1.vercodung]
        for c in (c1, c2):
            for col in ("connector_a", "connector_b"):
                c[col] = c[col].map(lambda x: shift_conn(x, k))
            c["von_bis"] = c.connector_a + "/" + c.pin_a + " - " + c.connector_b + "/" + c.pin_b
            c["wireid"] = c.wireid + (f"_c{k}" if k else "")
            c["part_number"] = c.part_number.map(lambda x: shift_pn(x, k))
            c["predecessor_pn"] = c.predecessor_pn.map(lambda x: shift_pn(x, k))
        P1.append(c1)
        P2.append(c2)
        v = key["VPC_Mapping"].copy()
        for col in v.columns:
            v[col] = v[col].map(lambda x: shift_conn(x, k))
        V.append(v)
        wk = wm.copy()
        sfx = f"_c{k}" if k else ""
        wk["p2_wireid"] = wk.p2_wireid + sfx
        wk["p1_wireid"] = wk.p1_wireid.map(lambda x: x + sfx if x else "")
        for col in ("p1_part_number", "p2_part_number"):
            wk[col] = wk[col].map(lambda x: shift_pn(x, k))
        W.append(wk)
        mk = key["Part_Number_Modules"].copy()
        for col in ("p1_part_number", "p2_part_number"):
            mk[col] = mk[col].map(lambda x: shift_pn(x, k))
        M.append(mk)
        dk = key["Dropped_from_P1"].copy()
        dk["p1_wireid"] = dk.p1_wireid + sfx
        Dr.append(dk)
        print(f"clone {k + 1}/{a.factor}")

    P1 = pd.concat(P1, ignore_index=True)
    P2 = pd.concat(P2, ignore_index=True)
    stem = f"x{a.factor}"
    write_wirelist(out / f"P1_{stem}.xlsx", P1, {"project": f"P1 bulk x{a.factor}", "kem": "BULK/P1"})
    write_wirelist(out / f"P2_{stem}.xlsx", P2, {"project": f"P2 bulk x{a.factor}", "kem": "BULK/P2"})
    s = key["Summary"].copy()
    s["Count"] = [str(int(c) * a.factor) if str(c).isdigit() else c for c in s.Count]
    write_key(out / f"KEY_{stem}.xlsx", {"Summary": s, "VPC_Mapping": pd.concat(V), "Wire_Mapping": pd.concat(W),
                                          "Part_Number_Modules": pd.concat(M), "Dropped_from_P1": pd.concat(Dr)})
    print(f"wrote {len(P1):,} / {len(P2):,} wires to {out}")


if __name__ == "__main__":
    main()
