# Harness Part-Number Commonizer

A local desktop app that compares DBX logical wirelists. Everything runs on this Mac: there are no cloud calls and no model downloads.

It does three things:

1. **Cross-project mapping**
   - Detects connector (VPC) renames.
   - Classifies each wire as carry-over, new, new-in-carried-module or dropped.
   - Maps each part number (module) from P1 to P2.
2. **Merge candidates.** Finds part numbers whose functions overlap. It uses connection overlap, harness_attribute and Vercodung logic, and returns a recommendation and a suggested combined code rule:
   - Absorb
   - Merge
   - Combine (same code rule)
   - Combine (shared connectors)
   - Variant pair
   - Review

   By default only part numbers with the **same Vercodung** are paired. "Same" means logically equal, so `A1A+(B2B/C3C)` matches `(C3C/B2B)+A1A`. Untick **Merge candidates: same Vercodung only** in the app, or pass `--any-vercodung`, to also see pairs with different code rules.
3. **Function clusters.** Groups similar part numbers using a similarity graph plus HDBSCAN. Each group gets a representative part, its core connections and the possible part-number reduction.
4. **Part number report.** One row per part number per project: wires, connectors, harness attribute, Vercodung, lineage to the other project, cluster, number of merge candidates and the best merge partner.

## Presentation

[`docs/Harness_Commonizer_Overview.pptx`](docs/Harness_Commonizer_Overview.pptx) is a 10-slide overview of how the system works. It has management slides (value, results, how to use it) and technical slides (methods, connector alignment, commonization, scale).

## Start

- **macOS:** double-click **`run_app.command`** in Finder.
- **Windows:** double-click **`run_app.bat`** in Explorer. Needs Python 3.9+ from python.org (with "tcl/tk" ticked, which is the default).

The first run creates `.venv` and installs the dependencies. On Windows, use `.venv\Scripts\python` wherever this README says `.venv/bin/python`. You can also start it from a terminal:

```bash
.venv/bin/python app.py
```

In the app:

1. **Add…** two or more wirelists, oldest first.
2. Optionally pick an answer key.
3. Click **Run analysis**.
4. Browse the tabs. You can filter, click a header to sort, double-click a row for details and page with ◀ ▶.
5. Use **Project:** above the tabs to show one project only. It applies to the tabs that have a project column (part number report, merge candidates, clusters, wires without part number). The P1→P2 mapping tabs always show both projects.
6. **Export view…** on any tab saves exactly the rows shown (project, filter and sort) to Excel. **Export to Excel…** saves the whole report, limited to the selected project if one is chosen.

## Command line (batch / very large files)

```bash
.venv/bin/python cli.py P1.xlsx P2.xlsx --key answer_key.xlsx -o report.xlsx
```

- `--mode chain|baseline` for more than 2 files.
- `--no-commonize` for mapping only.
- `--any-vercodung` to allow merge candidates with different Vercodung.
- `--project "NAME"` to limit the report to one project.
- `--config cfg.json` to override thresholds. The keys are the same as in **Settings…** in the app.

## How it works (classic ML, all local)

| Step | Method |
|---|---|
| Connector alignment | TF-IDF profile per connector (pin + stable wire fingerprint + far-end pin/connector), blocked by class letter and `.position`. One-to-one matching with the Hungarian algorithm (`scipy`). 2–3 propagation passes through the wiring graph, then a global pass for renames that left their block. |
| Wire matching | Exact key on translated endpoints + fingerprint (cross-section, colour, Leitungssachnummer, Kabelsachnummer, harness_attribute, long_name …). Fuzzy fallback with `rapidfuzz` and Hungarian assignment for wires whose attributes changed. |
| Module lineage | Majority vote over carried-over wires, cross-checked with `predecessor_part_number`. |
| Commonization | Weighted sparse module vectors (connections, Vercodung codes, harness, long_name words, wire profile). Blocked sparse-cosine kNN. Vercodung parser checks for conflicts (one module requires a code the other forbids). Clusters come from connected components, and HDBSCAN splits oversized ones (`scikit-learn`). |

Vercodung is re-coded between projects, so it is not used to match wires across projects. It is used to decide whether two modules can be combined.

## Results on the supplied synthetic data

| Check (vs answer key) | Result |
|---|---|
| VPC renames | 663/663 (precision and recall 1.0) |
| Wire carry-over pairs | 9,308/9,310. The 2 misses are parallel wires with identical endpoints and attributes, which cannot be told apart. |
| Wire status, touches_VPC | 100% |
| Module status and P1→P2 mapping | 100% (1,489 modules) |
| Run time | about 4 s |

**Bulk test.** `tools/scale_synth.py` builds an N× larger pair plus a matching key. At 10× (133,000 wires per project) accuracy is the same and the analysis takes about 70 s, using about 1.7 GB of RAM. Loaded files are cached as parquet in `~/.cache/harness_commonizer`, so re-runs skip the xlsx parsing.

```bash
.venv/bin/python tools/scale_synth.py P1.xlsx P2.xlsx KEY.xlsx --factor 10 --out-dir bulk
```

## Input robustness

- The wirelist sheet is found by its header, not its name.
- German and English column variants are mapped automatically.
- Connectors and pins are taken from `Von/Bis` when the separate columns are missing.
- Duplicate wire ids get a suffix.
- Missing optional columns are tolerated.
- csv input is also accepted.

## Tests

```bash
.venv/bin/python -m pytest -q tests
```
