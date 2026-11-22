"""Excel export.

Written row by row with xlsxwriter in constant-memory mode, so very large results do not
blow up RAM; sheets beyond Excel's row limit are split into _2, _3 … parts.
"""
from __future__ import annotations

import math
from typing import List

import numpy as np
import pandas as pd
import xlsxwriter

from .pipeline import Result

MAX_ROWS = 1_048_575
ORDER = ["Summary", "Evaluation", "VPC_Mapping", "Wire_Mapping", "Part_Number_Modules", "Dropped_from_P1",
         "Merge_Candidates", "Function_Clusters", "Cluster_Members", "Orphan_Wires_P2", "Evaluation_Mismatches"]
SCORE_COLS = {"confidence", "similarity", "connection_jaccard", "containment", "commonality", "score", "carry_share"}


def _cell(v):
    if v is None:
        return ""
    if isinstance(v, (frozenset, set, list, tuple)):
        return "; ".join(map(str, v))
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return "" if math.isnan(v) else float(v)
    return v


def _sheet_name(base: str, used: set) -> str:
    name = base[:31]
    i = 2
    while name in used:
        suffix = f"_{i}"
        name = base[: 31 - len(suffix)] + suffix
        i += 1
    used.add(name)
    return name


def write_report(results: List[Result], path: str) -> str:
    wb = xlsxwriter.Workbook(path, {"constant_memory": True, "strings_to_urls": False,
                                    "nan_inf_to_errors": True})
    head = wb.add_format({"bold": True, "bg_color": "#1F3A5F", "font_color": "#FFFFFF", "border": 1})
    used = set()
    multi = len(results) > 1
    try:
        for ri, res in enumerate(results):
            for name in ORDER + [k for k in res.tables if k not in ORDER]:
                df = res.tables.get(name)
                if df is None or (df.empty and name != "Summary"):
                    continue
                cols = list(df.columns)
                for start in range(0, max(len(df), 1), MAX_ROWS):
                    chunk = df.iloc[start:start + MAX_ROWS]
                    base = f"{ri + 1}_{name}" if multi else name
                    ws = wb.add_worksheet(_sheet_name(base, used))
                    for j, col in enumerate(cols):
                        sample = chunk[col].head(500).astype(str).str.len()
                        width = min(60, max(10, len(str(col)) + 2, int(sample.quantile(0.9)) + 2 if len(sample) else 10))
                        ws.set_column(j, j, width)
                    ws.write_row(0, 0, cols, head)
                    for i, row in enumerate(chunk.itertuples(index=False, name=None), start=1):
                        ws.write_row(i, 0, [_cell(v) for v in row])
                    n = len(chunk)
                    if n:
                        ws.autofilter(0, 0, n, len(cols) - 1)
                        for j, col in enumerate(cols):
                            if col in SCORE_COLS:
                                ws.conditional_format(1, j, n, j, {"type": "3_color_scale", "min_color": "#F8696B",
                                                                   "mid_color": "#FFEB84", "max_color": "#63BE7B"})
                    ws.freeze_panes(1, 0)
    finally:
        wb.close()
    return path
