"""Excel export.

Written row by row with xlsxwriter in constant-memory mode, so very large results do not
blow up RAM; sheets beyond Excel's row limit are split into _2, _3 … parts.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import xlsxwriter

from .pipeline import Result

MAX_ROWS = 1_048_575
ORDER = ["Summary", "Evaluation", "VPC_Mapping", "Wire_Mapping", "Part_Number_Modules", "Part_Number_Report", "Dropped_from_P1",
         "Merge_Candidates", "Function_Clusters", "Cluster_Members", "Orphan_Wires_P2", "Evaluation_Mismatches"]
SCORE_COLS = {"confidence", "similarity", "connection_jaccard", "containment", "commonality", "score", "carry_share"}

ALL_PROJECTS = "(all projects)"
PROJECT_COLS = ("project", "project_a", "project_b", "representative_project")


def project_scope(df: pd.DataFrame, project: str) -> pd.DataFrame:
    """Rows that belong to `project` (any project column matches). Tables without a project
    column (wire / VPC / lineage mapping between the two projects) are returned unchanged."""
    if not project or project == ALL_PROJECTS or df.empty:
        return df
    cols = [c for c in PROJECT_COLS if c in df.columns]
    if not cols and "projects" not in df.columns:
        return df
    mask = pd.Series(False, index=df.index)
    for c in cols:
        mask |= df[c].astype(str) == project
    if "projects" in df.columns:
        mask |= df["projects"].astype(str).str.split(", ").map(lambda xs: project in xs)
    return df[mask]


def has_project_column(df: pd.DataFrame) -> bool:
    return any(c in df.columns for c in PROJECT_COLS + ("projects",))


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


def _write_df(wb, head, ws_name: str, df: pd.DataFrame, used: set):
    cols = list(df.columns)
    for start in range(0, max(len(df), 1), MAX_ROWS):
        chunk = df.iloc[start:start + MAX_ROWS]
        ws = wb.add_worksheet(_sheet_name(ws_name, used))
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


def _workbook(path: str):
    wb = xlsxwriter.Workbook(path, {"constant_memory": True, "strings_to_urls": False,
                                    "nan_inf_to_errors": True})
    head = wb.add_format({"bold": True, "bg_color": "#1F3A5F", "font_color": "#FFFFFF", "border": 1})
    return wb, head


def write_tables(tables: Dict[str, pd.DataFrame], path: str) -> str:
    """Write {sheet name: DataFrame} as one workbook (used for 'Export view' of a single tab)."""
    wb, head = _workbook(path)
    used: set = set()
    try:
        for name, df in tables.items():
            _write_df(wb, head, name, df, used)
    finally:
        wb.close()
    return path


def write_report(results: List[Result], path: str,
                 table_filter: Optional[Callable[[str, pd.DataFrame], pd.DataFrame]] = None) -> str:
    """Full report. `table_filter(name, df)` may narrow each table (e.g. to one project)."""
    wb, head = _workbook(path)
    used: set = set()
    multi = len(results) > 1
    try:
        for ri, res in enumerate(results):
            for name in ORDER + [k for k in res.tables if k not in ORDER]:
                df = res.tables.get(name)
                if df is None:
                    continue
                if table_filter is not None:
                    df = table_filter(name, df)
                if df.empty and name != "Summary":
                    continue
                _write_df(wb, head, f"{ri + 1}_{name}" if multi else name, df, used)
    finally:
        wb.close()
    return path
