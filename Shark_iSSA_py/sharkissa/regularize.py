import numpy as np
import pandas as pd
import math

R_EARTH = 6371000.0

def _haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))

def _bearing(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1)*np.cos(lat2) - np.sin(lat1)*np.sin(lat2)*np.cos(dlon)
    ang = np.arctan2(y, x)
    return (ang + 2*np.pi) % (2*np.pi)

def _to_local_xy(lon: np.ndarray, lat: np.ndarray, lon0: float, lat0: float):
    lon_r = np.radians(lon); lat_r = np.radians(lat)
    lon0_r = math.radians(lon0); lat0_r = math.radians(lat0)
    x = R_EARTH * (lon_r - lon0_r) * math.cos(lat0_r)
    y = R_EARTH * (lat_r - lat0_r)
    return x, y

def regularize(df: pd.DataFrame, cfg) -> pd.DataFrame:
    g = df[[cfg.id_col, cfg.time_col, cfg.lon_col, cfg.lat_col]].dropna().copy()
    g[cfg.time_col] = pd.to_datetime(g[cfg.time_col], errors="coerce")
    g = g.dropna(subset=[cfg.time_col])
    lon0 = float(g[cfg.lon_col].median())
    lat0 = float(g[cfg.lat_col].median())

    out = []
    for sid, s in g.sort_values([cfg.id_col, cfg.time_col]).groupby(cfg.id_col):
        s = s.set_index(cfg.time_col).sort_index()
        s = s.resample(f"{cfg.dt_minutes}min").mean(numeric_only=True)
        s[cfg.lon_col] = s[cfg.lon_col].interpolate(limit_direction="both")
        s[cfg.lat_col] = s[cfg.lat_col].interpolate(limit_direction="both")
        s = s.reset_index()
        s[cfg.id_col] = sid

        s["lon_prev"] = s[cfg.lon_col].shift(1)
        s["lat_prev"] = s[cfg.lat_col].shift(1)

        s["step_len_m"] = _haversine(s["lon_prev"], s["lat_prev"], s[cfg.lon_col], s[cfg.lat_col])
        s["heading_rad"] = _bearing(s["lon_prev"], s["lat_prev"], s[cfg.lon_col], s[cfg.lat_col])
        s["heading_prev_rad"] = s["heading_rad"].shift(1)
        s["turn_rad"] = ((s["heading_rad"] - s["heading_prev_rad"] + np.pi) % (2*np.pi)) - np.pi
        s["speed_mps"] = s["step_len_m"] / (cfg.dt_minutes * 60.0)

        # Local XY for neighbor search
        x, y = _to_local_xy(s[cfg.lon_col].to_numpy(), s[cfg.lat_col].to_numpy(), lon0, lat0)
        s["x_m"] = x; s["y_m"] = y
        s["x_prev"] = s["x_m"].shift(1)
        s["y_prev"] = s["y_m"].shift(1)

        out.append(s)

    reg = pd.concat(out, ignore_index=True)

    # Carry sex (mode) if present
    if getattr(cfg, "sex_col", None) and cfg.sex_col in df.columns:
        sex_map = (df[[cfg.id_col, cfg.sex_col]].
                   dropna(subset=[cfg.id_col]).
                   groupby(cfg.id_col)[cfg.sex_col]
                   .agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0])
                   )
        reg = reg.merge(sex_map.rename(cfg.sex_col), left_on=cfg.id_col, right_index=True, how="left")

    # passthrough numeric columns (carried by nearest time per id)
    passthru = getattr(getattr(cfg, "covariates", None), "get", lambda *_: [])("passthrough", [])
    if passthru:
        keep = [c for c in passthru if c in df.columns and c not in [cfg.lon_col, cfg.lat_col, cfg.time_col]]
        if keep:
            aux = df[[cfg.id_col, cfg.time_col] + keep].copy()
            aux[cfg.time_col] = pd.to_datetime(aux[cfg.time_col], errors="coerce")
            # join by nearest time within same id
            reg = reg.merge(aux, on=[cfg.id_col, cfg.time_col], how="left")

    return reg.dropna(subset=["lon_prev","lat_prev","step_len_m","x_prev","y_prev"])
