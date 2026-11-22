import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from commonizer.config import Config  # noqa: E402
from commonizer.connector_align import align_connectors  # noqa: E402
from commonizer.io_loader import Wirelist, standardise  # noqa: E402
from commonizer.normalize import (combine_rules, connector_block, parse_connector,  # noqa: E402
                                  parse_vercodung, rules_conflict)
from commonizer.pipeline import compare  # noqa: E402


# ------------------------------------------------------------------ parsing
@pytest.mark.parametrize("name,expected", [
    ("A101X9.4", ("A", "A101X9", "4")),
    ("A101aX13.3", ("A", "A101aX13", "3")),
    ("X308a.2", ("X", "X308a", "2")),
    ("FA9X3", ("FA", "FA9X3", "")),
    ("W141sX8.4", ("W", "W141sX8", "4")),
])
def test_parse_connector(name, expected):
    assert parse_connector(name) == expected


def test_reference_points_are_not_vpcs():
    letter, base, pos = parse_connector("XR-E3M.1")
    assert letter.startswith("#")
    assert connector_block("A101X9.4") == "A|4"


def test_vercodung_parse_and_canonical():
    r = parse_vercodung("-(D6V/B0L)+-L0L+(J9D/O3A)+I0Q")
    assert r.positive == {"J9D", "O3A", "I0Q"}
    assert r.negative == {"D6V", "B0L", "L0L"}
    assert r.required == {"I0Q"}
    # order of terms / alternatives does not matter
    assert parse_vercodung("I0Q+(O3A/J9D)+-L0L+-(B0L/D6V)").canonical() == r.canonical()


def test_vercodung_conflict_and_combine():
    a, b = parse_vercodung("A1A+Z5X"), parse_vercodung("-A1A+Z5X")
    assert rules_conflict(a, b) == {"A1A"}
    c = parse_vercodung("B2B+Z5X")
    combined = combine_rules([a, c])
    assert combined.startswith("Z5X") and "A1A" in combined and "B2B" in combined
    assert parse_vercodung("").terms == ()
    assert parse_vercodung("((weird))+").terms  # degrades, never raises


def test_standardise_aliases_and_von_bis():
    df = pd.DataFrame({"WireID": ["w1"], "Von/Bis": ["X1.2/3 - A5X1.2/7"], "Querschnitt": ["0,50"],
                       "Leitungsfarbe": ["rt"]}, dtype="string")
    s = standardise(df)
    assert s.loc[0, "connector_a"] == "X1.2" and s.loc[0, "pin_b"] == "7"
    assert s.loc[0, "cross_section"] == "0.5"
    assert s.loc[0, "vercodung"] == ""  # missing optional columns are created


# ------------------------------------------------------------------ end to end on a tiny pair
def _wl(rows, name):
    cols = ["wireid", "connector_a", "pin_a", "connector_b", "pin_b", "cross_section", "color", "wire_pn",
            "harness_attribute", "vercodung", "part_number", "predecessor_pn", "long_name"]
    df = pd.DataFrame(rows, columns=cols, dtype="string")
    df["von_bis"] = df.connector_a + "/" + df.pin_a + " - " + df.connector_b + "/" + df.pin_b
    return Wirelist(name, name, standardise(df), {"project": name})


def test_rename_detection_and_wire_status():
    base = []
    for i in range(1, 9):
        base.append([f"a{i}", "A10X1.4", str(i), "X20", str(i), "0.5", ["rt", "bl", "gn"][i % 3], f"P{i}",
                     "DACH", "A1A", "M1", "-", f"Lamp {i}"])
    for i in range(1, 6):
        base.append([f"b{i}", "B30X2.2", str(i), "X20", str(10 + i), "0.75", "sw", f"Q{i}",
                     "DACH", "B2B", "M2", "-", f"Motor {i}"])
    p1 = _wl(base, "P1")
    p2rows = []
    for r in base:
        if r[0] == "b5":
            continue  # dropped
        r = list(r)
        r[0] = "n_" + r[0]
        r[1] = r[1].replace("A10X1.4", "A77cX9.4")  # VPC rename
        r[9] = "C3C" if r[10] == "M1" else "D4D"     # Vercodung re-coded
        r[11] = "M1" if r[10] == "M1" else "M2"
        r[10] = "N1" if r[10] == "M1" else "N2"
        p2rows.append(r)
    p2rows.append(["new1", "A77cX9.4", "20", "X20", "30", "1.5", "ws", "R1", "DACH", "C3C", "N1", "M1", "Fan"])
    p2 = _wl(p2rows, "P2")
    res = compare(p1, p2, Config())
    vpc = res.tables["VPC_Mapping"]
    assert list(zip(vpc.project1_connector, vpc.project2_connector)) == [("A10X1.4", "A77cX9.4")]
    wm = res.tables["Wire_Mapping"].set_index("p2_wireid")
    assert wm.loc["n_a3", "p1_wireid"] == "a3" and wm.loc["n_a3", "status"] == "carry-over"
    assert wm.loc["new1", "status"] == "new (added to carried module)"
    assert list(res.tables["Dropped_from_P1"].p1_wireid) == ["b5"]
    mods = res.tables["Part_Number_Modules"].set_index("p2_part_number")
    assert mods.loc["N1", "p1_part_number"] == "M1" and mods.loc["N1", "status"] == "carry-over"


def test_identity_kept_when_no_rename():
    rows = [[f"w{i}", "S5X1.3", str(i), "X9", str(i), "1", "rt", f"P{i}", "VORBAU", "A1A", "M", "-", "x"]
            for i in range(1, 6)]
    p1, p2 = _wl(rows, "P1"), _wl([["v" + r[0][1:]] + r[1:] for r in rows], "P2")
    _, mapping = align_connectors(*(compare.__globals__["_prep"](w.wires, Config()) for w in (p1, p2)), Config())
    assert mapping["S5X1.3"] == "S5X1.3"


# ------------------------------------------------------------------ same-Vercodung merge filter + project scope
def _mods_pair_rows():
    rows = []
    # M1 and M2: same connections, same code rule written in a different order -> candidate
    # M3: same connections, different rule -> only a candidate when the filter is off
    for pn, rule in (("M1", "A1A+(B2B/C3C)"), ("M2", "(C3C/B2B)+A1A"), ("M3", "D4D")):
        for i in range(1, 6):
            rows.append([f"{pn}w{i}", "S5X1.3", str(i), "X9", str(i), "1", "rt", f"P{i}", "VORBAU", rule, pn,
                         "-", "Lamp"])
    return rows


def test_merge_candidates_same_vercodung_only():
    from commonizer.commonize import build_modules, merge_candidates, module_matrix
    w = compare.__globals__["_prep"](_wl(_mods_pair_rows(), "P").wires, Config())
    m = build_modules(w, "P")
    on = Config()
    X, Xc = module_matrix(m, on)
    pairs = merge_candidates(m, X, on, Xc=Xc)
    assert {frozenset(p) for p in zip(pairs.part_number_a, pairs.part_number_b)} == {frozenset({"M1", "M2"})}
    off = Config(merge_same_vercodung_only=False)
    pairs = merge_candidates(m, X, off, Xc=Xc)
    assert len(pairs) == 3


def test_project_scope():
    from commonizer.report import ALL_PROJECTS, project_scope
    df = pd.DataFrame({"project_a": ["A", "A", "B"], "project_b": ["A", "B", "B"]})
    assert len(project_scope(df, "A")) == 2 and len(project_scope(df, ALL_PROJECTS)) == 3
    cl = pd.DataFrame({"projects": ["A, B", "B"]})
    assert len(project_scope(cl, "A")) == 1
    other = pd.DataFrame({"p1_wireid": ["x"]})
    assert len(project_scope(other, "A")) == 1  # no project column -> unchanged


def test_results_slides(tmp_path):
    from pptx import Presentation
    from commonizer.slides import write_slides
    base = [[f"a{i}", "A10X1.4", str(i), "X20", str(i), "0.5", "rt", f"P{i}", "DACH", "A1A", "M1", "-", "Lamp"]
            for i in range(1, 6)]
    p1 = _wl(base, "P1")
    p2 = _wl([["n" + r[0]] + [r[1].replace("A10X1.4", "A77cX9.4")] + r[2:10] + ["N1", "M1", "Lamp"] for r in base], "P2")
    for cfg in (Config(), Config(merge_same_vercodung_only=False)):
        res = compare(p1, p2, cfg)
        out = tmp_path / "s.pptx"
        write_slides([res, res], str(out), files=["a.xlsx", "b.xlsx"])
        prs = Presentation(str(out))
        text = " ".join(sh.text_frame.text for s in prs.slides for sh in s.shapes if sh.has_text_frame)
        assert "P1 → P2" in text and len(prs.slides) >= 7
