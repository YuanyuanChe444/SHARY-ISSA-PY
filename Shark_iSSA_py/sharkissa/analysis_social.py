import numpy as np, pandas as pd, math

def lead_follow_posthoc(reg: pd.DataFrame, cfg) -> pd.DataFrame:
    """For used steps only: compute whether neighbors appear behind at t+Δ."""
    out = []
    step = pd.Timedelta(minutes=cfg.dt_minutes)
    half = math.radians(cfg.cone_half_angle_deg)
    for (sid), g in reg.sort_values([cfg.id_col, cfg.time_col]).groupby(cfg.id_col, sort=False):
        for i in range(1, len(g)):
            t = g.iloc[i][cfg.time_col]
            tp = t + step
            # focal endpoint at t, bearing at t
            x_end, y_end = g.iloc[i]["x_m"], g.iloc[i]["y_m"]
            h = g.iloc[i]["heading_rad"]
            if not np.isfinite(h): 
                continue
            # all others at t+Δ
            others = reg[(reg[cfg.time_col]==tp) & (reg[cfg.id_col]!=sid)]
            if others.empty: 
                continue
            dx = others["x_m"].values - x_end
            dy = others["y_m"].values - y_end
            dist = np.hypot(dx, dy)
            in_r = dist <= cfg.neighbor_radius_m
            if not np.any(in_r): 
                lead_future = 0
            else:
                dx, dy = dx[in_r], dy[in_r]
                proj = dx*math.sin(h) + dy*math.cos(h)
                ang = np.arctan2(dx, dy)
                ad = np.abs(((ang - h + np.pi) % (2*np.pi)) - np.pi)
                behind = (proj < 0) & (ad <= half)
                lead_future = 1 if np.any(behind) else 0
            out.append({cfg.id_col: sid, cfg.time_col: t, "lead_future": lead_future})
    return pd.DataFrame(out)
