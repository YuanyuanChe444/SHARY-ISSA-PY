import numpy as np
import pandas as pd
from .covariates.social import TimeNeighborIndex


def _zscore(s: pd.Series) -> pd.Series:
    """Safe z-score with guard for std=0 or all-NaN."""
    if s is None or not np.issubdtype(s.dtype, np.number):
        return s
    m = np.nanmean(s)
    sd = np.nanstd(s)
    if not np.isfinite(sd) or sd == 0:
        return s - m
    return (s - m) / sd


def _get_passthrough_list(cfg) -> list:
    """Return list of passthrough covariate names; tolerate None/missing."""
    covs = getattr(cfg, "covariates", None)
    if covs is None:
        return []
    try:
        pts = covs.get("passthrough", [])
    except Exception:
        pts = []
    if pts is None:
        pts = []
    return [str(x) for x in pts if isinstance(x, (str, bytes))]


def add_social(design: pd.DataFrame, reg: pd.DataFrame, cfg) -> pd.DataFrame:
    """
    Enrich the used+available design with social covariates (endpoint time),
    add sex_M, standardize numerics, and sanitize NaN/Inf so statsmodels can fit.
    """
    # Ensure time dtype
    if not np.issubdtype(design["time"].dtype, np.datetime64):
        design["time"] = pd.to_datetime(design["time"], errors="coerce")
    tcol = getattr(cfg, "time_col")
    if tcol in reg.columns and not np.issubdtype(reg[tcol].dtype, np.datetime64):
        reg[tcol] = pd.to_datetime(reg[tcol], errors="coerce")

    # Build neighbor index
    ni = TimeNeighborIndex(cfg.neighbor_radius_m, cfg.cone_half_angle_deg)
    ni.build(reg, cfg.id_col, cfg.time_col)

    # Ensure social columns exist
    social_cols = [
        "nn_dist", "n_forward", "n_behind",
        "ahead_any", "behind_any",
        "mean_align_fwd", "rel_speed_fwd"
    ]
    for col in social_cols:
        if col not in design.columns:
            design[col] = np.nan

    n = len(design)
    print(f"[sharkissa] social: evaluating {n:,} endpoints…", flush=True)

    # Loop with itertuples for speed
    for i, r in enumerate(design.itertuples(index=True), start=0):
        if i and (i % 2000 == 0):
            print(f"[sharkissa] social progress: {i:,}/{n:,}", flush=True)

        vals = ni.query_social(
            getattr(r, "time"),
            float(getattr(r, "x_end")),
            float(getattr(r, "y_end")),
            float(getattr(r, "heading_step")),
            getattr(r, "id"),
        )
        for k, v in vals.items():
            design.at[r.Index, k] = v

    # ---------- Sex -> sex_M by id ----------
    if getattr(cfg, "sex_col", None) and cfg.sex_col in reg.columns:
        sex_map = (
            reg[[cfg.id_col, cfg.sex_col]]
            .dropna(subset=[cfg.id_col])
            .groupby(cfg.id_col)[cfg.sex_col]
            .agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0])
        )
        sexM_by_id = sex_map.astype(str).str.upper().str.startswith("M").astype(float)
        design["sex_M"] = design["id"].map(sexM_by_id).astype(float)
        design["sex_M"] = design["sex_M"].fillna(0.0)  # unknown → 0

    # ---------- Fill social NaNs deterministically ----------
    # Indicators/counts: NaN → 0
    for c in ["ahead_any", "behind_any", "n_forward", "n_behind"]:
        if c in design.columns:
            design[c] = design[c].fillna(0.0)

    # If no neighbors within radius, encode distance as "at the radius"
    if "nn_dist" in design.columns:
        no_neighbors = (
            design[["ahead_any", "behind_any", "n_forward", "n_behind"]]
            .fillna(0).sum(axis=1) == 0
        )
        design.loc[no_neighbors & design["nn_dist"].isna(), "nn_dist"] = float(cfg.neighbor_radius_m)
        # any remaining NaN (rare) → radius
        design["nn_dist"] = design["nn_dist"].fillna(float(cfg.neighbor_radius_m))

    # mean alignment / relative speed undefined when no forward neighbors → set 0 (neutral)
    for c in ["mean_align_fwd", "rel_speed_fwd"]:
        if c in design.columns:
            design[c] = design[c].fillna(0.0)

    # ---------- Drop strata with invalid cos_turn ----------
    if "cos_turn" in design.columns:
        bad = ~np.isfinite(design["cos_turn"])
        if bad.any():
            bad_strata = design.loc[bad, "stratum_id"].unique()
            print(f"[sharkissa] dropping {len(bad_strata):,} strata with invalid cos_turn", flush=True)
            design = design[~design["stratum_id"].isin(bad_strata)].copy()

    # ---------- Standardize numeric movement & social covariates ----------
    to_standardize = [
        c for c in ["log_l", "log_l2", "cos_turn", "nn_dist", "n_forward", "n_behind",
                    "mean_align_fwd", "rel_speed_fwd"]
        if c in design.columns and np.issubdtype(design[c].dtype, np.number)
    ]
    for c in to_standardize:
        design[c] = _zscore(design[c])

    # ---------- Standardize numeric passthrough covariates ----------
    for c in _get_passthrough_list(cfg):
        if c in design.columns and np.issubdtype(design[c].dtype, np.number):
            design[c] = _zscore(design[c])

    return design
