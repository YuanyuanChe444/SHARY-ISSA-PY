import os
import typer, yaml, pandas as pd
from ..config import Config
from ..regularize import regularize
from ..steps import build_used_available
from ..design import add_social
from ..fit.conditional import try_statsmodels_conditional_logit, export_for_r, summarize_conditional_logit
from ..validate import kfold_time_cv
from ..report import render_report
import json
import numpy as np
from sharkissa.clean import clean_raw

app = typer.Typer(add_completion=False)

@app.command()
def design(cfg: str):
    """Build iSSA design matrix from config YAML."""
    print("[sharkissa] reading config…", flush=True)
    with open(cfg, 'r') as f:
        y = yaml.safe_load(f)
    c = Config(**y['dataset'],
               dt_minutes=y['regularization']['dt_minutes'],
               K_available=y['steps']['K_available'],
               neighbor_radius_m=y['social']['radius_m'],
               cone_half_angle_deg=y['social']['cone_half_angle_deg'])
    print(f"[sharkissa] loading CSV: {c.path}", flush=True)
    raw = pd.read_csv(c.path)

    print("[sharkissa] cleaning raw data…", flush=True)
    clean, qc = clean_raw(raw, c)
    print("[sharkissa] cleaning summary:", qc, flush=True) # optional: save a cleaned file for provenance
    clean_out = y.get("output", {}).get("clean_csv", "outputs/cleaned.csv")
    if clean_out:
        os.makedirs(os.path.dirname(clean_out), exist_ok=True)
        clean.to_csv(clean_out, index=False)
        print(f"[sharkissa] wrote cleaned CSV → {clean_out}", flush=True)

    print("[sharkissa] regularizing tracks…", flush=True)
    reg = regularize(raw, c)
    print(f"[sharkissa] regularized rows: {len(reg):,} across {reg[c.id_col].nunique()} sharks", flush=True)

    print("[sharkissa] building used+available…", flush=True)
    dsg = build_used_available(reg, c)
    print(f"[sharkissa] strata rows (used+avail): {len(dsg):,}", flush=True)

    print("[sharkissa] computing social covariates (this is the slow step)…", flush=True)
    dsg = add_social(dsg, reg, c)

    out_csv = y['output'].get('design_csv', 'outputs/design.csv')
    if "/" in out_csv or "\\" in out_csv:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    dsg.to_csv(out_csv, index=False)
    print(f"[sharkissa] wrote design matrix → {out_csv} (rows={len(dsg):,}).", flush=True)

@app.command()
def fit(cfg: str):
    """Fit conditional logit; write report."""
    with open(cfg, 'r') as f:
        y = yaml.safe_load(f)
    formula_cols = y['fit']['formula']
    design_csv = y['output'].get('design_csv', 'outputs/design.csv')
    report_html = y['output'].get('report_html', 'outputs/report.html')

    dsg = pd.read_csv(design_csv)
    res = try_statsmodels_conditional_logit(dsg, formula_cols)



    # Keep only strata that are complete (no NaN/Inf) for the chosen formula
    cols = ['is_used','stratum_id'] + formula_cols
    missing_mask = dsg[formula_cols].isna().any(axis=1)
    inf_mask = ~np.isfinite(dsg[formula_cols].to_numpy()).all(axis=1)
    bad_rows = missing_mask | inf_mask
    if bad_rows.any():
        bad_strata = dsg.loc[bad_rows, 'stratum_id'].unique()
        n_bad_rows = int(bad_rows.sum())
        n_bad_strata = len(bad_strata)
        print(f"[sharkissa] dropping {n_bad_strata:,} strata ({n_bad_rows:,} rows) with NaN/Inf in {formula_cols}")
        dsg = dsg[~dsg['stratum_id'].isin(bad_strata)].copy()

    # Also drop any leftover NA in the minimal set
    dsg = dsg.dropna(subset=cols)

    fit_summary_txt, metrics = None, {}
    if res is not None:
        fit_summary_txt = str(res.summary())
        metrics.update(summarize_conditional_logit(res))
        # CV (may take time on huge sets; feel free to downsample for a first run)
        try:
            cv = kfold_time_cv(dsg, formula_cols, k=5)
            metrics.update(cv)
        except Exception as e:
            metrics["cv_error"] = str(e)
    else:
        export_for_r(dsg, formula_cols, out_csv=y['output'].get('r_export_csv','outputs/issa_design.csv'))

    out = render_report(design_csv, y, fit_summary_txt=fit_summary_txt, metrics=metrics, out_html=report_html)
    print(f"[sharkissa] report written → {out}")

if __name__ == "__main__":
    app()
