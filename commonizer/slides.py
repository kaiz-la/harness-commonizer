"""Results deck: one PowerPoint per run, built from the result tables.

Each slide stands on its own (comparison name and date in the header line), so slides can be
copied into another presentation. Charts are native PowerPoint charts, so they stay editable.
"""
from __future__ import annotations

import datetime as _dt
import numbers
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

from .pipeline import Result

# copper-on-graphite palette (same as docs/Harness_Commonizer_Overview.pptx)
INK = RGBColor(0x1E, 0x26, 0x2B)
INK2 = RGBColor(0x3A, 0x46, 0x50)
MUTED = RGBColor(0x8A, 0x97, 0xA0)
PANEL = RGBColor(0xF1, 0xF4, 0xF5)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
COPPER = RGBColor(0xC4, 0x6A, 0x2B)
SAND = RGBColor(0xE3, 0xB5, 0x8F)
TEAL = RGBColor(0x2E, 0x7D, 0x7A)
SLATE = RGBColor(0x5B, 0x6B, 0x75)
HEAD, BODY = "Cambria", "Calibri"
SLIDE_W, SLIDE_H = 13.333, 7.5

TableFilter = Optional[Callable[[str, pd.DataFrame], pd.DataFrame]]


# ---------------------------------------------------------------------------- primitives
def _text(slide, x, y, w, h, text, size=14, color=INK2, bold=False, font=BODY, align=PP_ALIGN.LEFT,
          anchor=MSO_ANCHOR.TOP, italic=False):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    lines = text if isinstance(text, list) else [text]
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        if i:
            p.space_before = Pt(6)
        r = p.add_run()
        r.text = str(line)
        f = r.font
        f.size, f.bold, f.italic, f.name = Pt(size), bold, italic, font
        f.color.rgb = color
    return tb


def _box(slide, x, y, w, h, fill, rounded=True):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE,
                                 Inches(x), Inches(y), Inches(w), Inches(h))
    if rounded:
        shp.adjustments[0] = 0.06
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    shp.line.fill.background()
    shp.shadow.inherit = False
    return shp


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else ""
    if isinstance(v, numbers.Integral):
        return f"{int(v):,}"
    if isinstance(v, numbers.Real):
        if pd.isna(v):
            return ""
        return f"{int(v):,}" if float(v).is_integer() and abs(v) >= 2 else f"{float(v):.2f}"
    return str(v)


def _truthy(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "x"])


def _plural(n: int, one: str, many: str) -> str:
    return f"{n:,} {one if n == 1 else many}"


def _clip(s: str, n: int) -> str:
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def _table(slide, df: pd.DataFrame, cols, x, y, w, widths, max_chars, size=11, row_h=0.36):
    """cols: list of (column, header). widths: inches per column."""
    n = len(df)
    shape = slide.shapes.add_table(n + 1, len(cols), Inches(x), Inches(y), Inches(w), Inches(row_h * (n + 1)))
    tbl = shape.table
    tbl.first_row = True
    for j, wd in enumerate(widths):
        tbl.columns[j].width = Inches(wd)
    for i in range(n + 1):
        tbl.rows[i].height = Inches(row_h)
        for j, (col, head) in enumerate(cols):
            cell = tbl.cell(i, j)
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.margin_top = cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            if i == 0:
                txt, color, bold = head, WHITE, True
                cell.fill.fore_color.rgb = INK
            else:
                v = df.iloc[i - 1][col]
                txt, color, bold = _clip(_fmt(v), max_chars[j]), (INK if j == 0 else INK2), j == 0
                cell.fill.fore_color.rgb = PANEL if i % 2 else WHITE
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.text = ""
            r = p.add_run()
            r.text = txt
            r.font.size, r.font.bold, r.font.name = Pt(size), bold, BODY
            r.font.color.rgb = color
    return shape


def _bar_chart(slide, x, y, w, h, cats, vals, title, color=TEAL, horizontal=True, number_format="#,##0"):
    data = CategoryChartData()
    data.categories = list(cats)
    data.add_series(title, [float(v) for v in vals])
    kind = XL_CHART_TYPE.BAR_CLUSTERED if horizontal else XL_CHART_TYPE.COLUMN_CLUSTERED
    gf = slide.shapes.add_chart(kind, Inches(x), Inches(y), Inches(w), Inches(h), data)
    ch = gf.chart
    ch.has_legend = False
    ch.has_title = True
    ch.chart_title.text_frame.text = title
    tp = ch.chart_title.text_frame.paragraphs[0]
    tp.runs[0].font.size, tp.runs[0].font.bold, tp.runs[0].font.name = Pt(13), False, BODY
    tp.runs[0].font.color.rgb = INK
    plot = ch.plots[0]
    plot.gap_width = 60 if len(vals) > 2 else 180
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.number_format, dl.number_format_is_linked = number_format, False
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    dl.font.size, dl.font.name = Pt(12), BODY
    dl.font.color.rgb = INK2
    ser = plot.series[0]
    ser.format.fill.solid()
    ser.format.fill.fore_color.rgb = color
    ca, va = ch.category_axis, ch.value_axis
    ca.tick_labels.font.size, ca.tick_labels.font.name = Pt(12), BODY
    ca.tick_labels.font.color.rgb = INK2
    ca.format.line.color.rgb = PANEL
    ca.has_major_gridlines = False
    if horizontal:
        ca.reverse_order = True
    va.visible = False
    va.has_major_gridlines = False
    va.minimum_scale = 0
    if len(vals):
        va.maximum_scale = max(float(max(vals)) * 1.2, 1)
    return gf


# ---------------------------------------------------------------------------- slide frame
class _Deck:
    def __init__(self, stamp: str):
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = Inches(SLIDE_W), Inches(SLIDE_H)
        self.blank = self.prs.slide_layouts[6]
        self.stamp = stamp
        self.n = 0

    def dark(self):
        s = self.prs.slides.add_slide(self.blank)
        s.background.fill.solid()
        s.background.fill.fore_color.rgb = INK
        return s

    def content(self, title: str, context: str, notes: str = ""):
        s = self.prs.slides.add_slide(self.blank)
        s.background.fill.solid()
        s.background.fill.fore_color.rgb = WHITE
        self.n += 1
        _text(s, 0.6, 0.35, 12.1, 0.3, context.upper(), size=11, color=COPPER, bold=True)
        _text(s, 0.6, 0.7, 12.1, 0.8, title, size=28, color=INK, bold=True, font=HEAD, anchor=MSO_ANCHOR.MIDDLE)
        _text(s, 0.6, 6.95, 9.0, 0.3, f"Harness Part-Number Commonizer · {self.stamp}", size=10, color=MUTED)
        if notes:
            s.notes_slide.notes_text_frame.text = notes
        return s

    def stat(self, s, x, y, w, h, value, label, dark=True, accent=False):
        _box(s, x, y, w, h, TEAL if accent else (INK if dark else PANEL))
        _text(s, x + 0.25, y + 0.25, w - 0.5, 0.8, value, size=34, bold=True, font=HEAD,
              color=WHITE if accent else (SAND if dark else INK))
        _text(s, x + 0.25, y + 1.05, w - 0.5, h - 1.2, label, size=14, color=WHITE if (dark or accent) else INK2)

    def empty_note(self, s, text, y=2.2):
        _box(s, 0.6, y, 12.1, 1.3, PANEL)
        _text(s, 0.9, y, 11.5, 1.3, text, size=16, color=INK2, anchor=MSO_ANCHOR.MIDDLE)


# ---------------------------------------------------------------------------- helpers on results
def _summary(res: Result) -> dict:
    s = res.tables.get("Summary")
    if s is None:
        return {}
    return {str(k).strip(): v for k, v in zip(s["Item"], s["Count"])}


def _pct(a, b) -> str:
    return f"{100 * a / b:.0f}%" if b else "–"


# ---------------------------------------------------------------------------- slides
def _key_figures(d: _Deck, res: Result, ctx: str, sm: dict):
    w2 = int(sm.get("Project 2 wires", 0))
    co = int(sm.get("Carry-over from Project 1", 0))
    new = int(sm.get("New connections", 0))
    drop = int(sm.get("Project 1 wires dropped", 0))
    vpc = int(sm.get("VPCs renamed (connector positions)", 0))
    mods = int(sm.get("Project 2 part numbers (modules)", 0))
    mco = int(sm.get("carried-over modules", 0))
    s = d.content(f"{_pct(co, w2)} of {res.target} wires carry over from {res.base}", ctx,
                  "Key figures of the comparison. Carry-over means the same physical connection exists in both "
                  "projects (after connector renames are taken into account).")
    stats = [(f"{w2:,}", f"wires in {res.target}", True, False),
             (f"{co:,}", f"carried over from {res.base} ({_pct(co, w2)})", True, True),
             (f"{new:,}", f"new connections ({_pct(new, w2)})", True, False),
             (f"{drop:,}", f"{res.base} wires dropped", True, False)]
    for i, (v, lab, dark, acc) in enumerate(stats):
        d.stat(s, 0.6 + i * 3.08, 1.75, 2.88, 2.2, v, lab, dark, acc)
    stats2 = [(f"{vpc:,}", "connector positions renamed (VPC)"),
              (f"{mods:,}", f"part numbers in {res.target}"),
              (f"{mco:,}", f"carried-over part numbers ({_pct(mco, mods)})"),
              (f"{mods - mco:,}", "new part numbers")]
    for i, (v, lab) in enumerate(stats2):
        d.stat(s, 0.6 + i * 3.08, 4.2, 2.88, 2.2, v, lab, dark=False)


def _wires(d: _Deck, res: Result, ctx: str, sm: dict):
    wm = res.tables.get("Wire_Mapping")
    if wm is None or wm.empty:
        return
    vc = wm.status.value_counts()
    drop = int(sm.get("Project 1 wires dropped", 0))
    cats = ["Carry-over", "New (added to carried module)", "New (new module)", f"Dropped from {res.base}"]
    vals = [int(vc.get("carry-over", 0)), int(vc.get("new (added to carried module)", 0)), int(vc.get("new", 0)), drop]
    s = d.content("Where the wires of the new project come from", ctx,
                  "Wire status per wire of the target project, plus wires of the base project that no longer exist.")
    _bar_chart(s, 0.5, 1.6, 7.2, 4.9, cats, vals, "Wires by status")
    touches = int(_truthy(wm.touches_VPC).sum()) if "touches_VPC" in wm else 0
    new_attr = [(k.split("'")[1], int(v)) for k, v in sm.items() if k.startswith("Rows with new harness attribute")]
    lines = [f"{touches:,} wires touch a renamed connector (VPC)."]
    if new_attr:
        lines.append("New harness attributes: " + ", ".join(f"{a} ({n:,} wires)" for a, n in new_attr) + ".")
    fuzzy = int((wm.get("match_method", pd.Series(dtype=str)).astype(str).str.contains("fuzzy")).sum())
    if fuzzy:
        lines.append(f"{fuzzy:,} carry-over wires matched with changed attributes (fuzzy match), worth a check.")
    _box(s, 8.1, 1.75, 4.6, 4.6, PANEL)
    _text(s, 8.4, 2.0, 4.0, 0.4, "Observations", size=17, bold=True, color=INK)
    _text(s, 8.4, 2.55, 4.0, 3.6, lines, size=14, color=INK2)


def _modules(d: _Deck, res: Result, ctx: str):
    m = res.tables.get("Part_Number_Modules")
    if m is None or m.empty:
        return
    vc = m.status.value_counts()
    co = m[m.status == "carry-over"]
    grown = int((pd.to_numeric(co.get("new_wires", 0), errors="coerce").fillna(0) > 0).sum())
    s = d.content("Part numbers: carried over, extended and new", ctx,
                  "Each part number of the target project is linked to its predecessor by majority vote of its "
                  "carried-over wires.")
    _bar_chart(s, 0.5, 1.6, 5.2, 3.0, ["Carried over", "  of which extended", "New"],
               [int(vc.get("carry-over", 0)), grown, int(vc.get("new", 0))], "Part numbers by status", color=COPPER)
    new = m[m.status == "new"].copy()
    new["p2_wire_count"] = pd.to_numeric(new.p2_wire_count, errors="coerce")
    top = new.sort_values("p2_wire_count", ascending=False).head(8)
    _text(s, 6.1, 1.65, 6.6, 0.4, "Largest new part numbers", size=16, bold=True, color=INK)
    if len(top):
        _table(s, top, [("p2_part_number", "Part number"), ("harness_attribute", "Harness attribute"),
                        ("Vercodung", "Vercodung"), ("p2_wire_count", "Wires")],
               6.1, 2.1, 6.6, [1.7, 1.9, 2.25, 0.75], [18, 22, 26, 8])
    else:
        _text(s, 6.1, 2.2, 6.6, 0.5, "No new part numbers.", size=14)
    ext = co.assign(new_wires=pd.to_numeric(co.new_wires, errors="coerce")).sort_values("new_wires", ascending=False)
    ext = ext[ext.new_wires > 0].head(4)
    if len(ext):
        _text(s, 0.6, 4.85, 5.2, 0.4, "Most extended carried-over part numbers", size=16, bold=True, color=INK)
        _table(s, ext, [("p2_part_number", "Part number"), ("p1_part_number", "Predecessor"),
                        ("new_wires", "New wires")], 0.6, 5.3, 5.2, [1.9, 1.9, 1.4], [18, 18, 8], row_h=0.3)


def _vpc(d: _Deck, res: Result, ctx: str, sm: dict):
    v = res.tables.get("VPC_Mapping")
    if v is None:
        return
    s = d.content(f"{len(v):,} connector positions were renamed", ctx,
                  "Connector renames are learned from the wiring around each connector (pins, wire attributes, "
                  "far-end connectors). Low confidence rows should be checked.")
    if v.empty:
        d.empty_note(s, "No connector renames found: connector names are the same in both projects.")
        return
    conf = pd.to_numeric(v.confidence, errors="coerce")
    low = int((conf < 0.5).sum())
    fam = int(sm.get("VPC families renamed", 0))
    d.stat(s, 0.6, 1.75, 2.9, 2.0, f"{len(v):,}", "positions renamed")
    d.stat(s, 0.6, 3.95, 2.9, 2.0, f"{fam:,}", "connector families renamed", dark=False)
    _text(s, 0.6, 6.15, 2.9, 0.6, f"{low:,} with confidence below 0.5", size=13, italic=True,
          color=COPPER if low else MUTED)
    vv = v.assign(_w=pd.to_numeric(v.get("p2_wire_ends", 0), errors="coerce")).sort_values("_w", ascending=False).head(10)
    vv = vv.assign(confidence=pd.to_numeric(vv.confidence, errors="coerce").map(lambda c: f"{c:.2f}"))
    _text(s, 3.9, 1.7, 8.8, 0.4, "Renames affecting the most wires", size=16, bold=True, color=INK)
    _table(s, vv, [("project1_connector", f"{_clip(res.base, 22)}"), ("project2_connector", f"{_clip(res.target, 22)}"),
                   ("_w", "Wire ends"), ("confidence", "Confidence")],
           3.9, 2.15, 8.8, [2.9, 2.9, 1.5, 1.5], [26, 26, 8, 6], row_h=0.38)


def _merge(d: _Deck, res: Result, ctx: str, rule: str):
    mc = res.tables.get("Merge_Candidates")
    if mc is None:
        return
    title = (f"{_plural(len(mc), 'pair', 'pairs')} of part numbers could be combined" if len(mc)
             else "No merge candidates under the current rule")
    s = d.content(title, ctx, f"Merge rule: {rule}. Candidates come from connection overlap, harness attribute and "
                              "Vercodung logic. Absorb = one part is almost fully contained in the other.")
    _text(s, 0.6, 1.55, 12.1, 0.35, f"Rule: {rule}", size=13, italic=True, color=MUTED)
    if mc.empty:
        d.empty_note(s, "No pair of part numbers passed the merge rule and the similarity threshold. "
                        "To see candidates with different code rules, untick 'Merge candidates: same Vercodung only' "
                        "and run again.", y=2.3)
        return
    vc = mc.recommendation.value_counts()
    _bar_chart(s, 0.5, 2.0, 4.6, 4.6, list(vc.index), list(vc.values), "Pairs by recommendation", color=COPPER)
    tag = {res.base: "1", res.target: "2"}
    top = mc.head(9).assign(a=lambda x: x.part_number_a + " [" + x.project_a.map(lambda p: tag.get(p, p)) + "]",
                            b=lambda x: x.part_number_b + " [" + x.project_b.map(lambda p: tag.get(p, p)) + "]",
                            sim=lambda x: x.similarity.map(lambda v: f"{v:.2f}"))
    _text(s, 5.4, 2.0, 7.3, 0.4, "Strongest candidates", size=16, bold=True, color=INK)
    _table(s, top, [("recommendation", "Recommendation"), ("a", "Part A (larger)"), ("b", "Part B"),
                    ("sim", "Similarity"), ("shared_connections", "Shared")],
           5.4, 2.45, 7.3, [2.3, 1.75, 1.75, 0.85, 0.65], [30, 22, 22, 5, 6], size=10, row_h=0.4)
    _text(s, 5.4, 2.55 + 0.4 * (len(top) + 1), 7.3, 0.3, f"[1] = {res.base}    [2] = {res.target}", size=11,
          color=MUTED)


def _clusters(d: _Deck, res: Result, ctx: str):
    cl = res.tables.get("Function_Clusters")
    if cl is None:
        return
    s = d.content(f"{len(cl):,} function groups, up to {int(cl.possible_reduction.sum()) if len(cl) else 0:,} "
                  "fewer part numbers", ctx,
                  "Function groups are connected sets of merge candidates. Possible reduction = members minus the "
                  "one representative part that would remain.")
    if cl.empty:
        d.empty_note(s, "No function groups: there are no linked merge candidates above the cluster threshold.")
        return
    d.stat(s, 0.6, 1.75, 2.9, 2.0, f"{len(cl):,}", "function groups (2+ parts)")
    d.stat(s, 0.6, 3.95, 2.9, 2.0, f"{int(cl.possible_reduction.sum()):,}", "possible part-number reduction", accent=True)
    top = cl.head(9).assign(commonality=lambda x: x.commonality.map(lambda v: f"{v:.2f}"))
    _text(s, 3.9, 1.7, 8.8, 0.4, "Largest groups", size=16, bold=True, color=INK)
    _table(s, top, [("cluster_id", "Group"), ("size", "Parts"), ("representative_part", "Representative"),
                    ("harness_attributes", "Harness attribute"), ("commonality", "Common core"),
                    ("possible_reduction", "Reduction")],
           3.9, 2.15, 8.8, [1.05, 0.7, 2.05, 2.6, 1.2, 1.2], [9, 5, 22, 30, 6, 6], size=11, row_h=0.4)


def _evaluation(d: _Deck, res: Result, ctx: str):
    ev = res.tables.get("Evaluation")
    if ev is None or ev.empty:
        return
    ev = ev[ev.area != "Summary counts"].assign(value=lambda x: pd.to_numeric(x.value, errors="coerce").map(
        lambda v: "" if pd.isna(v) else f"{v:.4f}"))
    s = d.content("Check against the answer key", ctx, "Only shown when an answer key was loaded for the run.")
    _table(s, ev, [("area", "Area"), ("metric", "Metric"), ("value", "Value"), ("detail", "Detail")],
           0.6, 1.7, 12.1, [3.2, 4.3, 1.8, 2.8], [30, 40, 8, 24], size=12, row_h=0.38)


# ---------------------------------------------------------------------------- entry point
def write_slides(results: List[Result], path: str, table_filter: TableFilter = None, project: str = "",
                 files: Optional[List[str]] = None) -> str:
    """Write a results deck. `table_filter(name, df)` narrows tables (e.g. to one project)."""
    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    d = _Deck(stamp)

    s = d.dark()
    _text(s, 0.8, 2.0, 11.5, 1.4, "Wirelist comparison results", size=44, bold=True, color=WHITE, font=HEAD,
          anchor=MSO_ANCHOR.BOTTOM)
    sub = [f"{r.base}  →  {r.target}" for r in results]
    _text(s, 0.8, 3.6, 11.5, 1.8, sub[:6] + (["…"] if len(sub) > 6 else []), size=20, color=SAND)
    meta = f"Generated {stamp}" + (f" · project filter: {project}" if project else "")
    if files:
        meta += " · files: " + ", ".join(_clip(f, 60) for f in files[:4])
    _text(s, 0.8, 6.2, 11.8, 0.8, meta, size=12, color=MUTED)

    for res in results:
        tables = res.tables
        if table_filter is not None:
            tables = {k: table_filter(k, v) for k, v in tables.items()}
        r = Result(res.base, res.target, tables, res.timings, getattr(res, "meta", {}))
        ctx = f"{r.base} → {r.target}" + (f" · {project}" if project else "")
        sm = _summary(res)
        rule = r.meta.get("merge_rule", "")
        _key_figures(d, r, ctx, sm)
        _wires(d, r, ctx, sm)
        _modules(d, r, ctx)
        _vpc(d, r, ctx, sm)
        if "Merge_Candidates" in r.tables:
            _merge(d, r, ctx, rule or "see Settings")
            _clusters(d, r, ctx)
        _evaluation(d, r, ctx)
    d.prs.save(path)
    return path
