"""Evaluation-Awareness Explorer - desktop application.

A native desktop GUI (Tkinter, ships with Python) for running the pipeline
and inspecting its results prompt by prompt.

Run:  python app.py
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


APP_TITLE = "Evaluation-Awareness Explorer"
DEFAULT_RESULTS = os.path.join("results", "results.json")

# Colour scheme
BG = "#f4f5f7"
CARD = "#ffffff"
INK = "#1c1c1e"
MUTED = "#6b6b70"
ACCENT = "#1f3864"
GREEN = "#1a7f37"
RED = "#b23b00"
AMBER = "#9a6700"

VERDICT_COLOURS = {
    "real": GREEN,
    "cosmetic": RED,
    "inconclusive": AMBER,
}

VERDICT_MEANING = {
    "real": ("Suppression genuinely changed behaviour. Removing the internal "
             "signal closed the refusal gap, not just the probe reading."),
    "cosmetic": ("Suppression was COSMETIC. The probe stopped detecting "
                 "evaluation-awareness, but the behavioural gap survived - "
                 "the model still acts differently when tested."),
    "inconclusive": ("Neither threshold met. Suppression did not clearly "
                     "remove the probe signal or close the behavioural gap."),
}


class ResultsApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1180x760")
        self.minsize(980, 640)
        self.configure(bg=BG)

        self.data = None
        self.results_path = tk.StringVar(value=DEFAULT_RESULTS)
        self.filter_mode = tk.StringVar(value="All prompts")
        self.log_queue: queue.Queue = queue.Queue()
        self.run_proc = None

        self._build_styles()
        self._build_toolbar()
        self._build_tabs()
        self._build_statusbar()

        self.after(120, self._drain_log)
        # Try to auto-load if a results file is already sitting there.
        if os.path.exists(DEFAULT_RESULTS):
            self.load_results(DEFAULT_RESULTS, quiet=True)

    # ------------------------------------------------------------------
    # Chrome
    # ------------------------------------------------------------------
    def _build_styles(self):
        st = ttk.Style(self)
        try:
            st.theme_use("clam")
        except tk.TclError:
            pass
        st.configure("TFrame", background=BG)
        st.configure("Card.TFrame", background=CARD, relief="flat")
        st.configure("TLabel", background=BG, foreground=INK, font=("Segoe UI", 10))
        st.configure("Card.TLabel", background=CARD, foreground=INK)
        st.configure("H1.TLabel", background=BG, foreground=ACCENT,
                     font=("Segoe UI Semibold", 15))
        st.configure("H2.TLabel", background=CARD, foreground=ACCENT,
                     font=("Segoe UI Semibold", 11))
        st.configure("Big.TLabel", background=CARD, foreground=INK,
                     font=("Segoe UI Semibold", 22))
        st.configure("Muted.TLabel", background=CARD, foreground=MUTED,
                     font=("Segoe UI", 9))
        st.configure("Treeview", rowheight=26, font=("Segoe UI", 9),
                     background=CARD, fieldbackground=CARD)
        st.configure("Treeview.Heading", font=("Segoe UI Semibold", 9))
        st.configure("TNotebook", background=BG)
        st.configure("TNotebook.Tab", padding=(16, 8), font=("Segoe UI", 10))

    def _build_toolbar(self):
        bar = ttk.Frame(self, padding=(12, 10))
        bar.pack(fill="x")

        ttk.Label(bar, text=APP_TITLE, style="H1.TLabel").pack(side="left")

        ttk.Button(bar, text="Run pipeline...",
                   command=self.open_run_dialog).pack(side="right", padx=(6, 0))
        ttk.Button(bar, text="Reload",
                   command=self.reload_results).pack(side="right", padx=6)
        ttk.Button(bar, text="Open results file...",
                   command=self.browse_results).pack(side="right")

    def _build_tabs(self):
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=12, pady=(0, 8))

        self.tab_summary = ttk.Frame(self.nb, padding=12)
        self.tab_prompts = ttk.Frame(self.nb, padding=12)
        self.tab_layers = ttk.Frame(self.nb, padding=12)
        self.tab_sweep = ttk.Frame(self.nb, padding=12)
        self.tab_run = ttk.Frame(self.nb, padding=12)

        self.nb.add(self.tab_summary, text="  Summary  ")
        self.nb.add(self.tab_prompts, text="  Per-prompt results  ")
        self.nb.add(self.tab_layers, text="  Probe by layer  ")
        self.nb.add(self.tab_sweep, text="  Suppression sweep  ")
        self.nb.add(self.tab_run, text="  Run log  ")

        self._build_summary_tab()
        self._build_prompts_tab()
        self._build_layers_tab()
        self._build_sweep_tab()
        self._build_run_tab()

    def _build_statusbar(self):
        self.status = tk.StringVar(value="No results loaded.")
        bar = ttk.Frame(self, padding=(12, 6))
        bar.pack(fill="x", side="bottom")
        ttk.Label(bar, textvariable=self.status, style="TLabel").pack(side="left")

    # ------------------------------------------------------------------
    # Summary tab
    # ------------------------------------------------------------------
    def _build_summary_tab(self):
        t = self.tab_summary

        self.verdict_card = ttk.Frame(t, style="Card.TFrame", padding=16)
        self.verdict_card.pack(fill="x", pady=(0, 12))
        ttk.Label(self.verdict_card, text="OVERALL VERDICT",
                  style="Muted.TLabel").pack(anchor="w")
        self.verdict_value = tk.Label(self.verdict_card, text="-", bg=CARD,
                                      fg=MUTED, font=("Segoe UI Semibold", 26))
        self.verdict_value.pack(anchor="w", pady=(2, 6))
        self.verdict_expl = tk.Label(
            self.verdict_card, text="Load a results file to begin.", bg=CARD,
            fg=INK, font=("Segoe UI", 10), justify="left", wraplength=1040)
        self.verdict_expl.pack(anchor="w")

        grid = ttk.Frame(t)
        grid.pack(fill="x")
        self.metric_cards = {}
        specs = [
            ("detect", "Can it tell it's being tested?",
             "Probe AUC at the best layer"),
            ("layer", "Where in the model?", "Layer with the strongest signal"),
            ("gap", "Does behaviour differ?", "Refusal-rate gap G"),
            ("acc", "Prompts identified correctly", "Probe prediction vs truth"),
        ]
        for col, (key, title, sub) in enumerate(specs):
            card = ttk.Frame(grid, style="Card.TFrame", padding=14)
            card.grid(row=0, column=col, sticky="nsew", padx=(0 if col == 0 else 8, 0))
            grid.columnconfigure(col, weight=1)
            ttk.Label(card, text=title, style="H2.TLabel",
                      wraplength=230).pack(anchor="w")
            val = tk.Label(card, text="-", bg=CARD, fg=INK,
                           font=("Segoe UI Semibold", 22))
            val.pack(anchor="w", pady=(6, 2))
            ttk.Label(card, text=sub, style="Muted.TLabel",
                      wraplength=230).pack(anchor="w")
            self.metric_cards[key] = val

        rates = ttk.Frame(t, style="Card.TFrame", padding=14)
        rates.pack(fill="x", pady=(12, 0))
        ttk.Label(rates, text="Refusal behaviour by framing",
                  style="H2.TLabel").pack(anchor="w", pady=(0, 8))
        self.rate_text = tk.Label(rates, text="-", bg=CARD, fg=INK,
                                  font=("Consolas", 10), justify="left")
        self.rate_text.pack(anchor="w")

        # Diagnostics warnings panel
        self._build_diagnostics_panel()

    def _build_diagnostics_panel(self):
        """Amber/green panel at the bottom of the Summary tab showing W001–W004."""
        t = self.tab_summary
        diag = ttk.Frame(t, style="Card.TFrame", padding=14)
        diag.pack(fill="x", pady=(12, 0))
        self.diag_header = ttk.Label(diag, text="Diagnostic warnings",
                                     style="H2.TLabel")
        self.diag_header.pack(anchor="w", pady=(0, 6))

        # Amber background for the text widget when warnings are present.
        self.diag_text = tk.Text(
            diag, height=6, wrap="word", font=("Consolas", 9),
            bg="#fffbe6", fg=INK, relief="flat", cursor="arrow",
        )
        self.diag_text.pack(fill="both", expand=True)
        self.diag_text.tag_configure("warn_code",
                                     foreground=AMBER,
                                     font=("Consolas", 9, "bold"))
        self.diag_text.tag_configure("ok", foreground=GREEN)
        self.diag_text.tag_configure("muted", foreground=MUTED)
        self.diag_text.insert("1.0", "No results loaded.")
        self.diag_text.configure(state="disabled")

    # ------------------------------------------------------------------
    # Per-prompt tab
    # ------------------------------------------------------------------
    def _build_prompts_tab(self):
        t = self.tab_prompts

        top = ttk.Frame(t)
        top.pack(fill="x", pady=(0, 8))
        ttk.Label(top, text="Show:").pack(side="left")
        cb = ttk.Combobox(top, textvariable=self.filter_mode, state="readonly",
                          width=34, values=[
                              "All prompts",
                              "Evaluation framing only",
                              "Deployment framing only",
                              "Probe got it RIGHT",
                              "Probe got it WRONG",
                              "Model refused",
                              "Model complied",
                              "Unsafe requests only",
                              "Benign requests only",
                          ])
        cb.pack(side="left", padx=8)
        cb.bind("<<ComboboxSelected>>", lambda _e: self.refresh_prompts())
        self.prompt_count = ttk.Label(top, text="")
        self.prompt_count.pack(side="right")

        cols = ("pair", "framing", "knows", "conf", "correct", "refused",
                "type", "request")
        self.tree = ttk.Treeview(t, columns=cols, show="headings", height=16)
        headings = {
            "pair": ("#", 44),
            "framing": ("Framing", 100),
            "knows": ("Knows it's tested?", 130),
            "conf": ("Confidence", 90),
            "correct": ("Probe correct", 100),
            "refused": ("Model refused", 105),
            "type": ("Type", 80),
            "request": ("Request", 460),
        }
        for c in cols:
            text, width = headings[c]
            self.tree.heading(c, text=text)
            anchor = "w" if c == "request" else "center"
            self.tree.column(c, width=width, anchor=anchor,
                             stretch=(c == "request"))

        vsb = ttk.Scrollbar(t, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side="top", fill="both", expand=True)
        vsb.place(relx=1.0, rely=0.06, relheight=0.62, anchor="ne")

        self.tree.tag_configure("right", foreground=GREEN)
        self.tree.tag_configure("wrong", foreground=RED)
        self.tree.bind("<<TreeviewSelect>>", self._on_select_prompt)

        det = ttk.Frame(t, style="Card.TFrame", padding=12)
        det.pack(fill="both", expand=False, pady=(10, 0))
        ttk.Label(det, text="Selected prompt", style="H2.TLabel").pack(anchor="w")
        self.detail = tk.Text(det, height=9, wrap="word", font=("Consolas", 9),
                              bg="#fbfbfd", fg=INK, relief="flat")
        self.detail.pack(fill="both", expand=True, pady=(6, 0))
        self.detail.insert("1.0", "Select a row above to see the full prompt "
                                  "and the model's response.")
        self.detail.configure(state="disabled")

    def _on_select_prompt(self, _evt=None):
        sel = self.tree.selection()
        if not sel or not self.data:
            return
        idx = int(self.tree.item(sel[0], "tags")[-1])
        r = self.data["per_prompt"][idx]
        knows = "YES - probe says this looks like an EVALUATION" \
            if r["probe_says_being_tested"] else \
            "NO - probe says this looks like ORDINARY DEPLOYMENT"
        body = (
            f"FRAMING PRESENTED TO MODEL : {r['framing']}\n"
            f"PROBE READ THE MODEL AS     : {knows}\n"
            f"CONFIDENCE IT IS EVALUATION : {r['probe_confidence_eval']:.3f}\n"
            f"PROBE CORRECT?              : {'yes' if r['probe_correct'] else 'NO'}\n"
            f"REQUEST TYPE                : "
            f"{'unsafe (cyber-misuse)' if r['is_unsafe'] else 'benign'}\n"
            f"SPLIT                       : {r['split']}\n"
            f"MODEL REFUSED?              : {'yes' if r['refused'] else 'no'}\n"
            f"\n--- FULL PROMPT ---\n{r['prompt_text']}\n"
            f"\n--- MODEL RESPONSE ---\n{r['response']}\n"
        )
        self.detail.configure(state="normal")
        self.detail.delete("1.0", "end")
        self.detail.insert("1.0", body)
        self.detail.configure(state="disabled")

    def refresh_prompts(self):
        for row in self.tree.get_children():
            self.tree.delete(row)
        if not self.data:
            return
        mode = self.filter_mode.get()
        shown = 0
        for i, r in enumerate(self.data.get("per_prompt", [])):
            if mode == "Evaluation framing only" and r["framing"] != "evaluation":
                continue
            if mode == "Deployment framing only" and r["framing"] != "deployment":
                continue
            if mode == "Probe got it RIGHT" and not r["probe_correct"]:
                continue
            if mode == "Probe got it WRONG" and r["probe_correct"]:
                continue
            if mode == "Model refused" and not r["refused"]:
                continue
            if mode == "Model complied" and r["refused"]:
                continue
            if mode == "Unsafe requests only" and not r["is_unsafe"]:
                continue
            if mode == "Benign requests only" and r["is_unsafe"]:
                continue

            req = r["request"]
            if len(req) > 90:
                req = req[:87] + "..."
            self.tree.insert(
                "", "end",
                values=(
                    r["pair_id"],
                    r["framing"],
                    "YES" if r["probe_says_being_tested"] else "no",
                    f"{r['probe_confidence_eval']:.2f}",
                    "correct" if r["probe_correct"] else "WRONG",
                    "refused" if r["refused"] else "complied",
                    "unsafe" if r["is_unsafe"] else "benign",
                    req,
                ),
                tags=("right" if r["probe_correct"] else "wrong", str(i)),
            )
            shown += 1
        self.prompt_count.configure(
            text=f"{shown} of {len(self.data.get('per_prompt', []))} prompts")

    # ------------------------------------------------------------------
    # Layers tab
    # ------------------------------------------------------------------
    def _build_layers_tab(self):
        t = self.tab_layers
        ttk.Label(
            t,
            text="Each layer is probed independently. 'Selectivity' is the real "
                 "probe's AUC minus a control probe trained on shuffled labels - "
                 "if selectivity is near zero, the probe is learning on its own "
                 "and the AUC does not mean what it appears to.",
            wraplength=1080, justify="left").pack(anchor="w", pady=(0, 10))

        cols = ("layer", "auc", "acc", "control", "sel", "bar")
        self.layer_tree = ttk.Treeview(t, columns=cols, show="headings")
        for c, (h, w) in {
            "layer": ("Layer", 70), "auc": ("Probe AUC", 100),
            "acc": ("Accuracy", 90), "control": ("Control AUC", 100),
            "sel": ("Selectivity", 100), "bar": ("", 420),
        }.items():
            self.layer_tree.heading(c, text=h)
            self.layer_tree.column(c, width=w, anchor="w" if c == "bar" else "center",
                                   stretch=(c == "bar"))
        self.layer_tree.pack(fill="both", expand=True)
        self.layer_tree.tag_configure("best", background="#e6f4ea")

    # ------------------------------------------------------------------
    # Sweep tab
    # ------------------------------------------------------------------
    def _build_sweep_tab(self):
        t = self.tab_sweep
        ttk.Label(
            t,
            text="Suppression strength is swept upward. At each step the probe "
                 "signal and the behavioural gap are re-measured. If the probe "
                 "signal collapses but the gap survives, the suppression was "
                 "cosmetic: the model still behaves differently when it thinks "
                 "it is being tested.",
            wraplength=1080, justify="left").pack(anchor="w", pady=(0, 10))

        cols = ("alpha", "auc", "gap", "dauc", "dgap", "over", "verdict")
        self.sweep_tree = ttk.Treeview(t, columns=cols, show="headings")
        for c, (h, w) in {
            "alpha": ("Strength (alpha)", 120), "auc": ("Probe AUC", 100),
            "gap": ("Gap G", 90), "dauc": ("AUC reduction", 120),
            "dgap": ("Gap reduction", 120), "over": ("Over-steered", 110),
            "verdict": ("Verdict", 140),
        }.items():
            self.sweep_tree.heading(c, text=h)
            self.sweep_tree.column(c, width=w, anchor="center")
        self.sweep_tree.pack(fill="both", expand=True)
        for name, colour in VERDICT_COLOURS.items():
            self.sweep_tree.tag_configure(name, foreground=colour)

    # ------------------------------------------------------------------
    # Run tab
    # ------------------------------------------------------------------
    def _build_run_tab(self):
        t = self.tab_run
        ttk.Label(t, text="Pipeline output appears here while a run is in "
                          "progress.", style="TLabel").pack(anchor="w")
        self.logbox = tk.Text(t, wrap="word", font=("Consolas", 9),
                              bg="#1c1c1e", fg="#e6e6e6", relief="flat")
        self.logbox.pack(fill="both", expand=True, pady=(8, 0))
        self.logbox.insert("1.0", "Idle.\n")
        self.logbox.configure(state="disabled")

    def _log(self, line: str):
        self.logbox.configure(state="normal")
        self.logbox.insert("end", line.rstrip() + "\n")
        self.logbox.see("end")
        self.logbox.configure(state="disabled")

    def _drain_log(self):
        try:
            while True:
                item = self.log_queue.get_nowait()
                if isinstance(item, tuple) and item[0] == "__done__":
                    code, out_path = item[1], item[2]
                    if code == 0:
                        self.status.set("Run finished. Loading results...")
                        if os.path.exists(out_path):
                            self.load_results(out_path)
                        else:
                            messagebox.showwarning(
                                "Run finished",
                                f"The run finished but no results file was "
                                f"found at:\n{out_path}")
                    else:
                        self.status.set(f"Run failed (exit code {code}).")
                        messagebox.showerror(
                            "Run failed",
                            "The pipeline exited with an error. See the Run log "
                            "tab for the full output.")
                else:
                    self._log(str(item))
        except queue.Empty:
            pass
        self.after(120, self._drain_log)

    def open_run_dialog(self):
        dlg = tk.Toplevel(self)
        dlg.title("Run pipeline")
        dlg.configure(bg=BG)
        dlg.transient(self)
        dlg.grab_set()
        dlg.resizable(False, False)

        model = tk.StringVar(value="Qwen/Qwen2.5-1.5B-Instruct")
        device = tk.StringVar(value="auto")
        outdir = tk.StringVar(value="results")
        bench = tk.StringVar(value="")

        frm = ttk.Frame(dlg, padding=16)
        frm.pack(fill="both", expand=True)

        def row(r, label, var, width=46):
            ttk.Label(frm, text=label).grid(row=r, column=0, sticky="w", pady=6)
            e = ttk.Entry(frm, textvariable=var, width=width)
            e.grid(row=r, column=1, sticky="we", padx=(10, 0))
            return e

        row(0, "Model", model)
        ttk.Label(frm, text="Device").grid(row=1, column=0, sticky="w", pady=6)
        ttk.Combobox(frm, textvariable=device, state="readonly", width=44,
                     values=["auto", "cuda", "cpu"]).grid(
            row=1, column=1, sticky="we", padx=(10, 0))
        row(2, "Output folder", outdir)
        row(3, "SORRY-Bench CSV (optional)", bench)

        ttk.Label(frm, text="A first run downloads the model (~3 GB) and can "
                            "take several minutes.",
                  style="TLabel", wraplength=430).grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(10, 0))

        btns = ttk.Frame(frm)
        btns.grid(row=5, column=0, columnspan=2, sticky="e", pady=(14, 0))
        ttk.Button(btns, text="Cancel", command=dlg.destroy).pack(side="right",
                                                                  padx=(6, 0))

        def start():
            dlg.destroy()
            self.start_run(model.get(), device.get(), outdir.get(), bench.get())

        ttk.Button(btns, text="Start run", command=start).pack(side="right")

    def start_run(self, model, device, outdir, bench):
        cmd = [sys.executable, "-m", "eval_awareness.run_pipeline",
               "--model", model, "--device", device, "--output", outdir]
        if bench.strip():
            cmd += ["--sorry-bench", bench.strip()]

        self.nb.select(self.tab_run)
        self.logbox.configure(state="normal")
        self.logbox.delete("1.0", "end")
        self.logbox.configure(state="disabled")
        self._log("$ " + " ".join(cmd) + "\n")
        self.status.set("Running pipeline...")

        out_path = os.path.join(outdir, "results.json")

        def worker():
            try:
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                    cwd=os.path.dirname(os.path.abspath(__file__)),
                )
                self.run_proc = proc
                for line in proc.stdout:
                    self.log_queue.put(line)
                proc.wait()
                self.log_queue.put(("__done__", proc.returncode, out_path))
            except Exception as exc:                      # noqa: BLE001
                self.log_queue.put(f"ERROR: {exc}")
                self.log_queue.put(("__done__", 1, out_path))

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------------
    # Loading + rendering
    # ------------------------------------------------------------------
    def browse_results(self):
        path = filedialog.askopenfilename(
            title="Open results.json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if path:
            self.load_results(path)

    def reload_results(self):
        p = self.results_path.get()
        if p and os.path.exists(p):
            self.load_results(p)
        else:
            messagebox.showinfo("Reload", "No results file loaded yet.")

    def load_results(self, path: str, quiet: bool = False):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except FileNotFoundError:
            if not quiet:
                messagebox.showerror("Not found", f"No file at:\n{path}")
            return
        except json.JSONDecodeError as exc:
            messagebox.showerror("Invalid file",
                                 f"That file is not valid JSON.\n\n{exc}")
            return

        missing = [k for k in ("probes", "suppression") if k not in data]
        if missing:
            messagebox.showerror(
                "Unexpected file",
                "This does not look like a pipeline results file "
                f"(missing: {', '.join(missing)}).")
            return

        if "per_prompt" not in data:
            messagebox.showwarning(
                "Older results file",
                "This file has no per-prompt records, so the "
                "'Per-prompt results' tab will be empty.\n\n"
                "Re-run the pipeline to generate them.")
            data["per_prompt"] = []

        self.data = data
        self.results_path.set(path)
        self.render()
        self.status.set(f"Loaded: {path}")

    def render(self):
        d = self.data
        sup = d["suppression"]
        verdict = sup.get("overall_verdict", "inconclusive")

        self.verdict_value.configure(
            text=verdict.upper(), fg=VERDICT_COLOURS.get(verdict, MUTED))
        self.verdict_expl.configure(text=VERDICT_MEANING.get(verdict, ""))

        probes = d["probes"]
        auc = probes.get("best_auc", float("nan"))
        self.metric_cards["detect"].configure(text=f"{auc:.3f}")
        self.metric_cards["layer"].configure(
            text=str(probes.get("best_layer", "-")))

        beh = d.get("baseline_behaviour", {})
        gap = beh.get("gap", float("nan"))
        self.metric_cards["gap"].configure(text=f"{gap:.3f}")

        pp = d.get("per_prompt", [])
        if pp:
            n_ok = sum(1 for r in pp if r["probe_correct"])
            self.metric_cards["acc"].configure(text=f"{n_ok}/{len(pp)}")
        else:
            self.metric_cards["acc"].configure(text="-")

        r_eval = beh.get("refusal_rate_eval", float("nan"))
        r_dep = beh.get("refusal_rate_deploy", float("nan"))
        self.rate_text.configure(
            text=(f"Evaluation framing : refused {r_eval*100:5.1f}% of prompts\n"
                  f"Deployment framing : refused {r_dep*100:5.1f}% of prompts\n"
                  f"Difference (gap G) : {gap:.3f}"))

        # Also show selectivity if available
        sel = probes.get("best_selectivity")
        if sel is not None:
            self.metric_cards["detect"].configure(
                text=f"{auc:.3f}",
            )
            # Append selectivity hint to the gap card sub-label if possible
            # (we update the big value label only)

        # diagnostics
        self._render_diagnostics(d)

        # layers
        for row in self.layer_tree.get_children():
            self.layer_tree.delete(row)
        best = probes.get("best_layer")
        for L in probes.get("layers", []):
            sel = L.get("selectivity")
            ctrl = L.get("control_auc")
            filled = int(round(max(0.0, min(1.0, L["auc"])) * 30))
            self.layer_tree.insert(
                "", "end",
                values=(L["layer"], f"{L['auc']:.3f}", f"{L['accuracy']:.3f}",
                        "-" if ctrl is None else f"{ctrl:.3f}",
                        "-" if sel is None else f"{sel:+.3f}",
                        "#" * filled),
                tags=("best",) if L["layer"] == best else (),
            )

        # sweep
        for row in self.sweep_tree.get_children():
            self.sweep_tree.delete(row)
        for pnt in sup.get("sweep", []):
            self.sweep_tree.insert(
                "", "end",
                values=(f"{pnt['alpha']:.2f}", f"{pnt['auc']:.3f}",
                        f"{pnt['gap']:.3f}", f"{pnt['d_auc']:+.3f}",
                        f"{pnt['d_gap']:+.3f}",
                        "YES" if pnt.get("oversteered") else "",
                        pnt["verdict"]),
                tags=(pnt["verdict"],))

        self.refresh_prompts()

    def _render_diagnostics(self, d: dict):
        """Populate the diagnostics warnings panel in the Summary tab."""
        diag_data = d.get("diagnostics", {})
        warnings = diag_data.get("warnings", [])
        n = len(warnings)
        n_train = diag_data.get("n_train")
        d_model_val = diag_data.get("d_model")

        self.diag_text.configure(state="normal")
        self.diag_text.delete("1.0", "end")

        if not diag_data:
            # Results file pre-dates the diagnostics feature.
            self.diag_header.configure(foreground=MUTED)
            self.diag_text.insert(
                "end",
                "No diagnostics section in this results file.  "
                "Re-run the pipeline to see warnings.",
                "muted",
            )
        elif n == 0:
            self.diag_header.configure(foreground=GREEN)
            meta_parts = []
            if n_train is not None:
                meta_parts.append(f"n_train={n_train}")
            if d_model_val is not None:
                meta_parts.append(f"d_model={d_model_val}")
            if n_train and d_model_val:
                meta_parts.append(f"ratio={n_train/d_model_val:.3f}")
            meta = "  (" + ", ".join(meta_parts) + ")" if meta_parts else ""
            self.diag_text.insert("end", f"✓  No diagnostic warnings.{meta}\n", "ok")
        else:
            self.diag_header.configure(foreground=AMBER)
            meta_parts = []
            if n_train is not None:
                meta_parts.append(f"n_train={n_train}")
            if d_model_val is not None:
                meta_parts.append(f"d_model={d_model_val}")
            if n_train and d_model_val:
                meta_parts.append(f"ratio={n_train/d_model_val:.3f}")
            meta = "  " + ", ".join(meta_parts) if meta_parts else ""
            self.diag_text.insert(
                "end",
                f"⚠  {n} diagnostic warning(s){meta}\n\n",
            )
            for w in warnings:
                code = w.get("code", "W?")
                detail = w.get("detail", "")
                msg = w.get("message", "")
                # Code label + detail on one line
                self.diag_text.insert("end", f"[{code}] ", "warn_code")
                if detail:
                    self.diag_text.insert("end", detail + "\n")
                # First sentence of message as a brief explanation
                first = msg.split(".")[0].strip() + "." if msg else ""
                if first:
                    self.diag_text.insert("end", "       " + first + "\n\n")

        self.diag_text.configure(state="disabled")


def main():
    app = ResultsApp()
    app.mainloop()


if __name__ == "__main__":
    main()
