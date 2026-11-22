#!/usr/bin/env python
"""Harness Part-Number Commonizer — local desktop app (Tkinter).

Everything runs on this machine: pandas + scikit-learn + scipy + rapidfuzz + networkx.
"""
from __future__ import annotations

import json
import queue
import threading
import traceback
from pathlib import Path
from tkinter import (BooleanVar, StringVar, Tk, Toplevel, filedialog, messagebox, ttk)
import tkinter as tk

import pandas as pd

from commonizer import __version__
from commonizer.config import Config
from commonizer.io_loader import load_answer_key, load_wirelist
from commonizer.pipeline import compare_many
from commonizer.report import (ALL_PROJECTS, PROJECT_COLS, project_scope, write_report,  # noqa: F401
                               write_tables)

PAGE = 1000
SETTINGS_FILE = Path.home() / ".harness_commonizer.json"
TAB_ORDER = ["Summary", "Evaluation", "VPC_Mapping", "Wire_Mapping", "Part_Number_Modules", "Part_Number_Report", "Dropped_from_P1",
             "Merge_Candidates", "Function_Clusters", "Cluster_Members", "Orphan_Wires_P2", "Evaluation_Mismatches"]
TAB_TITLES = {"VPC_Mapping": "VPC renames", "Wire_Mapping": "Wires", "Part_Number_Modules": "Part numbers",
              "Dropped_from_P1": "Dropped", "Merge_Candidates": "Merge candidates",
              "Function_Clusters": "Function clusters", "Cluster_Members": "Cluster members",
              "Orphan_Wires_P2": "Wires w/o part no.", "Evaluation_Mismatches": "Key mismatches",
              "Part_Number_Report": "Part number report"}



class TableView(ttk.Frame):
    """Paged, filterable, sortable view of a DataFrame (keeps the UI fast for 100k+ rows)."""

    def __init__(self, master, df: pd.DataFrame, name: str = "table", project: str = ALL_PROJECTS):
        super().__init__(master)
        self.name = name
        self.df_src = df.reset_index(drop=True)
        self.project = project
        self.df_all = project_scope(self.df_src, project)
        self.df = self.df_all
        self.page = 0
        self.sort_col, self.sort_asc = None, True

        bar = ttk.Frame(self)
        bar.pack(fill="x", pady=(4, 2))
        ttk.Label(bar, text="Filter:").pack(side="left")
        self.q = StringVar()
        e = ttk.Entry(bar, textvariable=self.q, width=40)
        e.pack(side="left", padx=4)
        e.bind("<Return>", lambda _: self.apply_filter())
        self.col = StringVar(value="(all columns)")
        ttk.Combobox(bar, textvariable=self.col, values=["(all columns)"] + list(df.columns), width=26,
                     state="readonly").pack(side="left")
        ttk.Button(bar, text="Apply", command=self.apply_filter).pack(side="left", padx=4)
        ttk.Button(bar, text="Clear", command=self.clear_filter).pack(side="left")
        ttk.Button(bar, text="Export view…", command=self.export_view).pack(side="left", padx=(12, 0))
        self.info = ttk.Label(bar, text="")
        self.info.pack(side="right")
        ttk.Button(bar, text="▶", width=3, command=lambda: self.goto(self.page + 1)).pack(side="right")
        ttk.Button(bar, text="◀", width=3, command=lambda: self.goto(self.page - 1)).pack(side="right")

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True)
        cols = [str(c) for c in df.columns]
        self.tree = ttk.Treeview(body, columns=cols, show="headings", selectmode="browse")
        for c in cols:
            self.tree.heading(c, text=c, command=lambda c=c: self.sort(c))
            sample = df[c].head(200).astype(str).str.len()
            w = min(420, max(70, 8 * max(len(c), int(sample.quantile(0.9)) if len(sample) else 8)))
            self.tree.column(c, width=w, stretch=False)
        vs = ttk.Scrollbar(body, orient="vertical", command=self.tree.yview)
        hs = ttk.Scrollbar(body, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vs.set, xscrollcommand=hs.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vs.grid(row=0, column=1, sticky="ns")
        hs.grid(row=1, column=0, sticky="ew")
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=1)
        self.tree.bind("<Double-1>", self.show_row)
        self.tree.tag_configure("odd", background="#F3F6FA")
        self.render()

    def apply_filter(self):
        q = self.q.get().strip()
        if not q:
            self.df = self.df_all
        else:
            if self.col.get() == "(all columns)":
                text = self.df_all.astype(str).agg(" ".join, axis=1)
            else:
                text = self.df_all[self.col.get()].astype(str)
            self.df = self.df_all[text.str.contains(q, case=False, regex=False)]
        self.page = 0
        self.render()

    def set_project(self, project: str):
        self.project = project
        self.df_all = project_scope(self.df_src, project)
        self.apply_filter()

    def export_view(self):
        """Export exactly the rows shown (project scope + text filter + sort) to its own workbook."""
        scope = "" if self.project == ALL_PROJECTS else "_" + "".join(ch if ch.isalnum() else "_" for ch in self.project)
        p = filedialog.asksaveasfilename(defaultextension=".xlsx", initialfile=f"{self.name}{scope}.xlsx",
                                         filetypes=[("Excel", "*.xlsx")], parent=self)
        if not p:
            return
        try:
            write_tables({self.name: self.df}, p)
            messagebox.showinfo("Exported", f"{len(self.df):,} rows written to\n{p}", parent=self)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Export failed", str(e), parent=self)

    def clear_filter(self):
        self.q.set("")
        self.apply_filter()

    def sort(self, c):
        self.sort_asc = not self.sort_asc if self.sort_col == c else True
        self.sort_col = c
        key = pd.to_numeric(self.df[c], errors="coerce")
        if key.notna().mean() > 0.9:
            self.df = self.df.assign(_k=key).sort_values("_k", ascending=self.sort_asc).drop(columns="_k")
        else:
            self.df = self.df.sort_values(c, ascending=self.sort_asc, key=lambda s: s.astype(str))
        self.page = 0
        self.render()

    def goto(self, p):
        last = max(0, (len(self.df) - 1) // PAGE)
        self.page = min(max(0, p), last)
        self.render()

    def render(self):
        self.tree.delete(*self.tree.get_children())
        start = self.page * PAGE
        chunk = self.df.iloc[start:start + PAGE]
        for i, row in enumerate(chunk.itertuples(index=False, name=None)):
            vals = ["; ".join(map(str, v)) if isinstance(v, (set, frozenset, list, tuple)) else v for v in row]
            self.tree.insert("", "end", values=vals, tags=("odd",) if i % 2 else ())
        n = len(self.df)
        self.info.config(text=f"rows {start + 1 if n else 0:,}–{min(start + PAGE, n):,} of {n:,}"
                              + (f"  (filtered from {len(self.df_all):,})" if n != len(self.df_all) else ""))

    def show_row(self, _):
        sel = self.tree.selection()
        if not sel:
            return
        vals = self.tree.item(sel[0], "values")
        top = Toplevel(self)
        top.title("Row details")
        txt = tk.Text(top, wrap="word", width=100, height=30, font=("Menlo", 11))
        for c, v in zip(self.tree["columns"], vals):
            txt.insert("end", f"{c}: ", "k")
            txt.insert("end", f"{v}\n")
        txt.tag_configure("k", font=("Menlo", 11, "bold"))
        txt.config(state="disabled")
        txt.pack(fill="both", expand=True)


class SettingsDialog(Toplevel):
    def __init__(self, master, cfg: Config, on_save):
        super().__init__(master)
        self.title("Settings — thresholds & weights")
        self.on_save = on_save
        self.vars = {}
        frm = ttk.Frame(self, padding=10)
        frm.pack(fill="both", expand=True)
        for i, (k, v) in enumerate(cfg.to_dict().items()):
            ttk.Label(frm, text=k).grid(row=i, column=0, sticky="w", pady=1)
            var = StringVar(value=json.dumps(v) if isinstance(v, list) else str(v))
            ttk.Entry(frm, textvariable=var, width=70 if isinstance(v, list) else 12).grid(row=i, column=1, sticky="w")
            self.vars[k] = (var, type(v))
        b = ttk.Frame(frm)
        b.grid(row=len(self.vars), column=0, columnspan=2, pady=8)
        ttk.Button(b, text="Reset defaults", command=self.reset).pack(side="left", padx=4)
        ttk.Button(b, text="Save", command=self.save).pack(side="left", padx=4)

    def reset(self):
        for k, v in Config().to_dict().items():
            self.vars[k][0].set(json.dumps(v) if isinstance(v, list) else str(v))

    def save(self):
        try:
            d = {}
            for k, (var, typ) in self.vars.items():
                s = var.get()
                if typ is list:
                    d[k] = json.loads(s)
                elif typ is bool:
                    if s.strip().lower() not in ("true", "false", "1", "0", "yes", "no"):
                        raise ValueError(f"{k}: use True or False")
                    d[k] = s.strip().lower() in ("true", "1", "yes")
                else:
                    d[k] = typ(s)
            self.on_save(Config.from_dict(d))
            self.destroy()
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Invalid value", str(e), parent=self)


class App:
    def __init__(self, root: Tk):
        self.root = root
        root.title(f"Harness Part-Number Commonizer {__version__} — local")
        root.geometry("1400x880")
        self.q: queue.Queue = queue.Queue()
        self.results = []
        self.cfg = self.load_cfg()
        self.files = []
        self.views = []

        style = ttk.Style()
        if "aqua" not in style.theme_names():
            style.theme_use("clam")
        style.configure("Treeview", rowheight=22)

        top = ttk.Frame(root, padding=8)
        top.pack(fill="x")

        lf = ttk.LabelFrame(top, text="1  Wirelists (oldest → newest)", padding=6)
        lf.pack(side="left", fill="both", expand=True)
        self.lb = tk.Listbox(lf, height=5, selectmode="extended", activestyle="none")
        self.lb.pack(side="left", fill="both", expand=True)
        bb = ttk.Frame(lf)
        bb.pack(side="left", padx=6)
        ttk.Button(bb, text="Add…", command=self.add_files).pack(fill="x")
        ttk.Button(bb, text="Remove", command=self.remove_files).pack(fill="x", pady=2)
        ttk.Button(bb, text="Up", command=lambda: self.move(-1)).pack(fill="x")
        ttk.Button(bb, text="Down", command=lambda: self.move(1)).pack(fill="x", pady=2)

        of = ttk.LabelFrame(top, text="2  Options", padding=6)
        of.pack(side="left", fill="y", padx=8)
        self.mode = StringVar(value="chain")
        ttk.Radiobutton(of, text="Chain (1→2, 2→3 …)", variable=self.mode, value="chain").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(of, text="Baseline → each", variable=self.mode, value="baseline").grid(row=1, column=0, sticky="w")
        self.baseline = ttk.Combobox(of, values=[], width=28, state="readonly")
        self.baseline.grid(row=1, column=1, padx=4)
        self.do_common = BooleanVar(value=True)
        ttk.Checkbutton(of, text="Commonization (merge candidates + clusters)", variable=self.do_common
                        ).grid(row=2, column=0, columnspan=2, sticky="w")
        self.same_ver = BooleanVar(value=self.cfg.merge_same_vercodung_only)
        ttk.Checkbutton(of, text="Merge candidates: same Vercodung only", variable=self.same_ver,
                        command=self.toggle_same_ver).grid(row=3, column=0, columnspan=2, sticky="w")
        ttk.Label(of, text="Answer key (optional):").grid(row=4, column=0, sticky="w", pady=(6, 0))
        self.key = StringVar()
        kf = ttk.Frame(of)
        kf.grid(row=5, column=0, columnspan=2, sticky="we")
        ttk.Entry(kf, textvariable=self.key, width=40).pack(side="left")
        ttk.Button(kf, text="…", width=3, command=self.pick_key).pack(side="left")
        ttk.Button(kf, text="×", width=3, command=lambda: self.key.set("")).pack(side="left")
        ttk.Button(of, text="Settings…", command=self.open_settings).grid(row=6, column=0, sticky="w", pady=6)

        rf = ttk.LabelFrame(top, text="3  Run", padding=6)
        rf.pack(side="left", fill="y")
        self.run_btn = ttk.Button(rf, text="▶  Run analysis", command=self.run)
        self.run_btn.pack(fill="x")
        self.export_btn = ttk.Button(rf, text="Export to Excel…", command=self.export, state="disabled")
        self.export_btn.pack(fill="x", pady=4)
        self.prog = ttk.Progressbar(rf, length=220, mode="determinate", maximum=1.0)
        self.prog.pack(fill="x", pady=4)
        self.status = ttk.Label(rf, text="Add two or more wirelists", wraplength=230)
        self.status.pack(fill="x")

        pw = ttk.PanedWindow(root, orient="vertical")
        pw.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        res = ttk.Frame(pw)
        sel = ttk.Frame(res)
        sel.pack(fill="x")
        ttk.Label(sel, text="Comparison:").pack(side="left")
        self.cmp = ttk.Combobox(sel, state="readonly", width=70)
        self.cmp.pack(side="left", padx=4)
        self.cmp.bind("<<ComboboxSelected>>", lambda _: self.show_result(self.cmp.current()))
        ttk.Label(sel, text="Project:").pack(side="left", padx=(16, 0))
        self.proj = ttk.Combobox(sel, state="readonly", width=34, values=[ALL_PROJECTS])
        self.proj.set(ALL_PROJECTS)
        self.proj.pack(side="left", padx=4)
        self.proj.bind("<<ComboboxSelected>>", lambda _: self.apply_project())
        self.proj_note = ttk.Label(sel, text="", foreground="#5A6B80")
        self.proj_note.pack(side="left", padx=6)
        self.nb = ttk.Notebook(res)
        self.nb.pack(fill="both", expand=True)
        pw.add(res, weight=5)
        logf = ttk.Frame(pw)
        self.log = tk.Text(logf, height=8, font=("Menlo", 11), state="disabled", background="#0F1720",
                           foreground="#D7E3F4")
        self.log.pack(fill="both", expand=True)
        pw.add(logf, weight=1)
        self.root.after(100, self.poll)

    # ---------------------------------------------------------------- files
    def add_files(self):
        paths = filedialog.askopenfilenames(title="Select wirelists",
                                            filetypes=[("Wirelists", "*.xlsx *.xlsm *.csv"), ("All", "*.*")])
        for p in paths:
            if p not in self.files:
                self.files.append(p)
        self.refresh_files()

    def remove_files(self):
        for i in sorted(self.lb.curselection(), reverse=True):
            del self.files[i]
        self.refresh_files()

    def move(self, d):
        s = self.lb.curselection()
        if len(s) != 1:
            return
        i, j = s[0], s[0] + d
        if 0 <= j < len(self.files):
            self.files[i], self.files[j] = self.files[j], self.files[i]
            self.refresh_files()
            self.lb.selection_set(j)

    def refresh_files(self):
        self.lb.delete(0, "end")
        names = [Path(p).name for p in self.files]
        for i, n in enumerate(names):
            self.lb.insert("end", f"{i + 1}. {n}")
        self.baseline["values"] = names
        if names and not self.baseline.get():
            self.baseline.current(0)
        self.status.config(text=f"{len(names)} file(s) ready" if len(names) >= 2 else "Add two or more wirelists")

    def pick_key(self):
        p = filedialog.askopenfilename(title="Answer key", filetypes=[("Excel", "*.xlsx")])
        if p:
            self.key.set(p)

    # ---------------------------------------------------------------- settings
    def load_cfg(self) -> Config:
        try:
            return Config.from_dict({**Config().to_dict(), **json.loads(SETTINGS_FILE.read_text())})
        except Exception:  # noqa: BLE001
            return Config()

    def save_cfg(self):
        try:
            SETTINGS_FILE.write_text(json.dumps(self.cfg.to_dict(), indent=1))
        except OSError:
            pass

    def toggle_same_ver(self):
        self.cfg.merge_same_vercodung_only = bool(self.same_ver.get())
        self.save_cfg()

    def open_settings(self):
        def save(cfg):
            self.cfg = cfg
            self.same_ver.set(cfg.merge_same_vercodung_only)
            try:
                SETTINGS_FILE.write_text(json.dumps(cfg.to_dict(), indent=1))
            except OSError:
                pass
        SettingsDialog(self.root, self.cfg, save)

    # ---------------------------------------------------------------- run
    def write_log(self, msg):
        self.log.config(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.config(state="disabled")

    def run(self):
        if len(self.files) < 2:
            messagebox.showinfo("Need files", "Add at least two wirelists.")
            return
        self.run_btn.config(state="disabled")
        self.export_btn.config(state="disabled")
        self.prog["value"] = 0
        args = dict(files=list(self.files), key=self.key.get().strip(), mode=self.mode.get(),
                    baseline=max(0, self.baseline.current()), commonize=self.do_common.get(), cfg=self.cfg)
        threading.Thread(target=self.worker, kwargs=args, daemon=True).start()

    def worker(self, files, key, mode, baseline, commonize, cfg):
        try:
            prog = lambda msg, frac=None: self.q.put(("progress", msg, frac))  # noqa: E731
            lists = []
            for i, p in enumerate(files):
                lists.append(load_wirelist(p, lambda m: prog(m)))
            k = load_answer_key(key) if key else None
            res = compare_many(lists, mode=mode, baseline=baseline, cfg=cfg, key=k, progress=prog,
                               commonize=commonize)
            self.q.put(("done", res, None))
        except Exception as e:  # noqa: BLE001
            self.q.put(("error", f"{e}\n\n{traceback.format_exc()}", None))

    def poll(self):
        try:
            while True:
                kind, a, b = self.q.get_nowait()
                if kind == "progress":
                    self.write_log(a)
                    if b is not None:
                        self.prog["value"] = b
                        self.status.config(text=a)
                elif kind == "done":
                    self.results = a
                    self.cmp["values"] = [f"{i + 1}: {r.base}  →  {r.target}" for i, r in enumerate(a)]
                    self.cmp.current(0)
                    self.show_result(0)
                    self.run_btn.config(state="normal")
                    self.export_btn.config(state="normal")
                    self.status.config(text="Finished — browse the tabs or export")
                elif kind == "exported":
                    self.export_btn.config(state="normal")
                    self.status.config(text=f"Saved {Path(a).name}")
                    self.write_log(f"Report written to {a}")
                elif kind == "error":
                    self.run_btn.config(state="normal")
                    self.export_btn.config(state="normal" if self.results else "disabled")
                    self.write_log(a)
                    messagebox.showerror("Error", a.split("\n")[0])
        except queue.Empty:
            pass
        self.root.after(100, self.poll)

    def projects_of(self, r):
        found = []
        for df in r.tables.values():
            for c in PROJECT_COLS:
                if c in df.columns:
                    found += [x for x in df[c].astype(str).unique() if x and x not in found]
        return sorted(found)

    def tab_title(self, view: "TableView"):
        title = TAB_TITLES.get(view.name, view.name.replace("_", " "))
        return title if view.name == "Summary" else f"{title} ({len(view.df_all):,})"

    def apply_project(self):
        p = self.proj.get() or ALL_PROJECTS
        for tab_id, view in self.views:
            view.set_project(p)
            self.nb.tab(tab_id, text=self.tab_title(view))
        self.proj_note.config(text="" if p == ALL_PROJECTS else
                              "filters tabs with a project column; P1→P2 mapping tabs show both projects")

    def show_result(self, idx):
        for t in self.nb.tabs():
            self.nb.forget(t)
        self.views = []
        r = self.results[idx]
        projects = self.projects_of(r)
        self.proj["values"] = [ALL_PROJECTS] + projects
        if self.proj.get() not in projects:
            self.proj.set(ALL_PROJECTS)
        for name in TAB_ORDER + [k for k in r.tables if k not in TAB_ORDER]:
            df = r.tables.get(name)
            if df is None or (df.empty and name != "Summary"):
                continue
            view = TableView(self.nb, df, name, self.proj.get())
            self.nb.add(view, text=self.tab_title(view))
            self.views.append((self.nb.tabs()[-1], view))
        self.apply_project()
        tm = ", ".join(f"{k} {v:.1f}s" for k, v in r.timings.items())
        self.write_log(f"Timings: {tm}")

    def export(self):
        if not self.results:
            return
        proj = self.proj.get() or ALL_PROJECTS
        scope = "" if proj == ALL_PROJECTS else "_" + "".join(ch if ch.isalnum() else "_" for ch in proj)
        p = filedialog.asksaveasfilename(defaultextension=".xlsx", initialfile=f"commonizer_report{scope}.xlsx",
                                         filetypes=[("Excel", "*.xlsx")])
        if not p:
            return
        flt = None if proj == ALL_PROJECTS else (lambda name, df: project_scope(df, proj))
        self.export_btn.config(state="disabled")
        self.status.config(text="Writing Excel…")

        def job():
            try:
                write_report(self.results, p, flt)
                self.q.put(("exported", p, None))
            except Exception as e:  # noqa: BLE001
                self.q.put(("error", f"Export failed: {e}", None))
        threading.Thread(target=job, daemon=True).start()


def main():
    root = Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
