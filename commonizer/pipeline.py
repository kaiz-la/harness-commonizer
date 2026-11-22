"""End-to-end orchestration with progress callbacks (used by both CLI and GUI)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import pandas as pd

from .commonize import (build_modules, function_clusters, merge_candidates, module_matrix,
                        orphan_suggestions, part_number_report)
from .config import Config
from .connector_align import align_connectors
from .evaluate import evaluate
from .io_loader import Wirelist
from .module_map import map_modules
from .normalize import add_endpoint_columns, wire_fingerprint
from .wire_match import match_wires, translate

Progress = Callable[[str, Optional[float]], None]


@dataclass
class Result:
    base: str
    target: str
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    timings: Dict[str, float] = field(default_factory=dict)
    meta: Dict[str, str] = field(default_factory=dict)


def _prep(w: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    w = add_endpoint_columns(w)
    w["fp"] = wire_fingerprint(w, cfg.fingerprint_cols)
    return w


def project_labels(base: Wirelist, target: Wirelist):
    a, b = base.label or "Project 1", target.label or "Project 2"
    return (a + " (1)", b + " (2)") if a == b else (a, b)


def compare(base: Wirelist, target: Wirelist, cfg: Optional[Config] = None,
            key: Optional[Dict[str, pd.DataFrame]] = None, progress: Optional[Progress] = None,
            commonize: bool = True) -> Result:
    cfg = cfg or Config()
    say = (lambda msg, frac=None: progress(msg, frac)) if progress else (lambda msg, frac=None: None)
    log = lambda msg: say(msg, None)  # noqa: E731
    res = Result(base.label, target.label)
    res.meta["merge_rule"] = ("same Vercodung only" + (" (empty Vercodung included)" if cfg.merge_include_empty_vercodung
                                                       else "") if cfg.merge_same_vercodung_only
                              else "any Vercodung (conflicts flagged)")
    T = time.perf_counter
    t0 = T()

    say(f"Preparing {base.label} → {target.label}", 0.02)
    w1, w2 = _prep(base.wires, cfg), _prep(target.wires, cfg)

    say("Aligning connectors (VPC renames)…", 0.08)
    t = T()
    vpc, mapping = align_connectors(w1, w2, cfg, log)
    res.timings["connector alignment"] = T() - t

    say("Matching wires…", 0.30)
    t = T()
    wire_map, dropped = match_wires(w1, w2, mapping, cfg, log)
    res.timings["wire matching"] = T() - t

    say("Mapping part numbers…", 0.45)
    t = T()
    modules = map_modules(wire_map, w1, w2)
    res.timings["module mapping"] = T() - t

    carried_mods = set(modules.loc[modules.status == "carry-over", "p2_part_number"])
    added = (wire_map.status == "new") & wire_map.p2_part_number.isin(carried_mods)
    wire_map.loc[added, "status"] = "new (added to carried module)"
    n_co = int((wire_map.status == "carry-over").sum())
    n_new = int(wire_map.status.str.startswith("new").sum())
    new_in_carried = int(added.sum())
    summary = [
        ("Project 2 wires", len(w2)),
        ("Carry-over from Project 1", n_co),
        ("New connections", n_new),
        ("  of which added to carried-over modules", new_in_carried),
        ("Project 1 part numbers (modules)", int(w1.loc[w1.part_number != "", "part_number"].nunique())),
        ("  P1 modules carried into Project 2 (lineage)",
         int(modules.loc[modules.status == "carry-over", "p1_part_number"].nunique())),
        ("Project 2 part numbers (modules)", len(modules)),
        ("  carried-over modules", int((modules.status == "carry-over").sum())),
        ("  new modules", int((modules.status == "new").sum())),
        ("Project 1 wires dropped", len(dropped)),
        ("VPCs renamed (connector positions)", len(vpc)),
        ("VPC families renamed", int(vpc[["base_p1", "base_p2"]].drop_duplicates().shape[0]) if len(vpc) else 0),
    ]
    for attr in sorted(set(w2.harness_attribute) - set(w1.harness_attribute)):
        if attr:
            summary.append((f"Rows with new harness attribute '{attr}'", int((w2.harness_attribute == attr).sum())))
    res.tables["Summary"] = pd.DataFrame(summary, columns=["Item", "Count"])
    res.tables["VPC_Mapping"] = vpc
    res.tables["Wire_Mapping"] = wire_map
    res.tables["Part_Number_Modules"] = modules
    res.tables["Dropped_from_P1"] = dropped

    if commonize:
        say("Building module feature vectors…", 0.55)
        t = T()
        tt = T()
        t1 = translate(w1, mapping)  # project-1 wires in project-2 connector space
        lab1, lab2 = project_labels(base, target)
        mods = pd.concat([build_modules(t1, lab1), build_modules(w2, lab2)], ignore_index=True)
        X, Xc = module_matrix(mods, cfg)
        log(f"  {len(mods):,} modules × {X.shape[1]:,} features ({T() - tt:.1f}s)")
        tt = T()
        # carry-over lineage is not commonization: drop P1/P2 pairs linked by carried wires or declaration
        carried = wire_map[wire_map.status == "carry-over"]
        lineage = set(zip(carried.p2_part_number, carried.p1_part_number))
        lineage |= set(zip(modules.p2_part_number, modules.p1_part_number))
        say("Finding merge candidates…", 0.65)
        pairs = merge_candidates(mods, X, cfg, exclude_pairs=lineage, progress=log, Xc=Xc)
        log(f"  merge candidates scored ({T() - tt:.1f}s)")
        tt = T()
        say("Clustering functions…", 0.80)
        clusters, members = function_clusters(mods, X, pairs, cfg, log)
        log(f"  clustering done ({T() - tt:.1f}s)")
        tt = T()
        say("Suggesting modules for wires without part number…", 0.90)
        orphans = orphan_suggestions(w2, mods, X, lab2, cfg)
        if len(orphans):
            orphans.insert(0, "project", lab2)
        log(f"  {len(orphans):,} orphan wires scored ({T() - tt:.1f}s)")
        res.timings["commonization"] = T() - t
        res.tables["Merge_Candidates"] = pairs
        res.tables["Function_Clusters"] = clusters
        res.tables["Cluster_Members"] = members
        res.tables["Orphan_Wires_P2"] = orphans
        res.tables["Part_Number_Report"] = part_number_report(mods, modules, lab1, lab2, pairs, members)
        if len(pairs):
            rec = pairs.recommendation.value_counts()
            for k, v in rec.items():
                res.tables["Summary"].loc[len(res.tables["Summary"])] = (f"Merge candidates: {k}", int(v))
        if len(clusters):
            res.tables["Summary"].loc[len(res.tables["Summary"])] = ("Function clusters (≥2 parts)", len(clusters))
            res.tables["Summary"].loc[len(res.tables["Summary"])] = (
                "  possible part-number reduction", int(clusters.possible_reduction.sum()))

    if key:
        say("Scoring against answer key…", 0.95)
        ev, mism = evaluate(res.tables, key)
        res.tables["Evaluation"] = ev
        res.tables["Evaluation_Mismatches"] = mism
    res.timings["total"] = T() - t0
    say(f"Done in {res.timings['total']:.1f}s", 1.0)
    return res


def compare_many(lists: List[Wirelist], mode: str = "chain", baseline: int = 0, **kw) -> List[Result]:
    """chain: 1→2, 2→3, …   baseline: baseline→each other."""
    if len(lists) < 2:
        raise ValueError("need at least two wirelists")
    if mode == "baseline":
        pairs = [(lists[baseline], l) for i, l in enumerate(lists) if i != baseline]
    else:
        pairs = list(zip(lists[:-1], lists[1:]))
    return [compare(a, b, **kw) for a, b in pairs]
