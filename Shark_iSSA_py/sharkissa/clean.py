# sharkissa/clean.py
from __future__ import annotations
import numpy as np
import pandas as pd

EARTH_RADIUS_M = 6_371_008.8


def _haversine(lon1, lat1, lon2, lat2):
    """Vectorized haversine distance in meters."""
    lon1 = np.radians(lon1); lat1 = np.radians(lat1)
    lon2 = np.radians(lon2); lat2 = np.radians(lat2)
    dlon = lon2 - lon1; dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def _cfg_get(cfg, path, default=None):
    """
    Safe getter for nested config values.
    path like "cleaning.max_speed_m_s".
    Works whether cfg has attributes or a dict.
    """
    cur = cfg
    for part in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part, default if part == path.split(".")[-1] else {})
        else:
            cur = getattr(cur, part, default if part == path.split(".")[-1] else {})
    return cur if cur is not None else default


def clean_raw(df: pd.DataFrame, cfg) -> tuple[pd.DataFrame, dict]:
    """
    Returns (cleaned_df, qc_summary).
    Cleaning steps are conservative & configurable; nothing “future-looking.”
    """
    id_col  = cfg.id_col
    time_col= cfg.time_col
    lon_col = cfg.lon_col
    lat_col = cfg.lat_col

    # ---- defaults / knobs (all overrideable via YAML cleaning: ...) ----
    drop_zero_zero     = bool(_cfg_get(cfg, "cleaning.drop_zero_zero", True))
    bounds             = _cfg_get(cfg, "cleaning.bounds", None)  # [min_lon, min_lat, max_lon, max_lat] or None
    max_speed_m_s      = float(_cfg_get(cfg, "cleaning.max_speed_m_s", 6.0))  # be generous
    min_points_per_id  = int(_cfg_get(cfg, "cleaning.min_points_per_id", 10))
    min_segment_points = int(_cfg_get(cfg, "cleaning.min_segment_points", 3))
    max_gap_minutes    = float(_cfg_get(cfg, "cleaning.max_gap_minutes", 480.0))  # 8 hours → split segments
    prefer_low_HPE     = bool(_cfg_get(cfg, "cleaning.prefer_low_HPE_on_duplicates", True))

    qc = {}

    # ---- 0) Keep necessary columns & cast types ----
    g = df.copy()
    need = [id_col, time_col, lon_col, lat_col]
    # required present?
    missing = [c for c in need if c not in g.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns {missing}. Available: {list(g.columns)}")

    # coerce types
    g[time_col] = pd.to_datetime(g[time_col], errors="coerce")
    g[lon_col]  = pd.to_numeric(g[lon_col], errors="coerce")
    g[lat_col]  = pd.to_numeric(g[lat_col], errors="coerce")

    # ---- 1) Drop rows missing id/time/lon/lat ----
    before = len(g)
    g = g.dropna(subset=[id_col, time_col, lon_col, lat_col])
    qc["dropped_missing_core"] = before - len(g)

    # ---- 2) Range checks + (0,0) ----
    before = len(g)
    rng_mask = g[lat_col].between(-90, 90) & g[lon_col].between(-180, 180)
    if drop_zero_zero:
        rng_mask &= ~((g[lon_col] == 0) & (g[lat_col] == 0))
    g = g[rng_mask]
    qc["dropped_out_of_range"] = before - len(g)

    # Optional bounding box
    if bounds and len(bounds) == 4:
        min_lon, min_lat, max_lon, max_lat = bounds
        before = len(g)
        bb = g[lon_col].between(min_lon, max_lon) & g[lat_col].between(min_lat, max_lat)
        g = g[bb]
        qc["dropped_outside_bounds"] = before - len(g)

    # ---- 3) De-duplicate id+time (keep best HPE if available) ----
    before = len(g)
    if prefer_low_HPE and "HPE" in g.columns:
        g = g.sort_values(["HPE"])  # lower HPE first
        g = g.drop_duplicates(subset=[id_col, time_col], keep="first")
    else:
        g = g.drop_duplicates(subset=[id_col, time_col], keep="first")
    qc["dropped_duplicates"] = before - len(g)

    # ---- 4) Sort & compute dt/step/speed for filters ----
    g = g.sort_values([id_col, time_col]).reset_index(drop=True)
    g["__lon_prev"] = g.groupby(id_col)[lon_col].shift()
    g["__lat_prev"] = g.groupby(id_col)[lat_col].shift()
    g["__time_prev"] = g.groupby(id_col)[time_col].shift()

    # negative/zero dt → bad ordering or duplicates we missed
    g["__dt_s"] = (g[time_col] - g["__time_prev"]).dt.total_seconds()
    # first per-id row has NaN dt/step/speed
    g["__step_m"] = _haversine(g["__lon_prev"], g["__lat_prev"], g[lon_col], g[lat_col])
    g["__speed_m_s"] = g["__step_m"] / g["__dt_s"]

    # ---- 5) Split segments on big gaps; drop tiny segments ----
    # new segment when gap > max_gap_minutes or dt <= 0
    big_gap = (g["__dt_s"] > max_gap_minutes * 60) | (g["__dt_s"] <= 0) | g["__dt_s"].isna()
    g["__segment"] = big_gap.groupby(g[id_col]).cumsum()

    # Drop segments with too few points (need ≥ min_segment_points to define steps cleanly)
    before = len(g)
    seg_sizes = g.groupby([id_col, "__segment"]).size()
    keep_segs = seg_sizes[seg_sizes >= min_segment_points].index
    g = g.set_index([id_col, "__segment"]).loc[keep_segs].reset_index()
    qc["dropped_small_segments"] = before - len(g)

    # recompute dt/speed within kept segments
    g = g.sort_values([id_col, "__segment", time_col]).reset_index(drop=True)
    g["__lon_prev"]  = g.groupby([id_col, "__segment"])[lon_col].shift()
    g["__lat_prev"]  = g.groupby([id_col, "__segment"])[lat_col].shift()
    g["__time_prev"] = g.groupby([id_col, "__segment"])[time_col].shift()
    g["__dt_s"]      = (g[time_col] - g["__time_prev"]).dt.total_seconds()
    g["__step_m"]    = _haversine(g["__lon_prev"], g["__lat_prev"], g[lon_col], g[lat_col])
    g["__speed_m_s"] = g["__step_m"] / g["__dt_s"]

    # ---- 6) Remove impossible speeds ----
    before = len(g)
    bad_speed = (g["__speed_m_s"] > max_speed_m_s) & np.isfinite(g["__speed_m_s"])
    g = g[~bad_speed].copy()
    qc["dropped_speed_outliers"] = before - len(g)

    # ---- 7) Drop IDs with too-few points overall ----
    before = len(g)
    id_sizes = g.groupby(id_col).size()
    keep_ids = id_sizes[id_sizes >= min_points_per_id].index
    g = g[g[id_col].isin(keep_ids)].copy()
    qc["dropped_ids_too_small"] = before - len(g)

    # Cleanup helper columns
    g = g.drop(columns=[c for c in g.columns if c.startswith("__")], errors="ignore")

    # Final sorting
    g = g.sort_values([id_col, time_col]).reset_index(drop=True)

    # Return cleaned df + summary
    qc["final_rows"] = len(g)
    qc["final_ids"]  = g[id_col].nunique()
    return g, qc
