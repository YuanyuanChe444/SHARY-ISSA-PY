import numpy as np
import pandas as pd

class AvailableSampler:
    def __init__(self, K=20, seed=42):
        self.K = K
        self.rng = np.random.default_rng(seed)

    def sample(self, steps: pd.DataFrame) -> pd.DataFrame:
        pool_len = steps["step_len_m"].dropna().values
        pool_turn = steps["turn_rad"].dropna().values
        if len(pool_len) == 0 or len(pool_turn) == 0:
            raise ValueError("Not enough steps to sample available steps.")
        rows = []
        for idx, r in steps.iterrows():
            if not np.isfinite(r.get("heading_rad", np.nan)):
                continue
            lengths = self.rng.choice(pool_len, size=self.K, replace=True)
            turns   = self.rng.choice(pool_turn, size=self.K, replace=True)
            h0 = r["heading_prev_rad"] if np.isfinite(r.get("heading_prev_rad", np.nan)) else r["heading_rad"]
            headings = (h0 + turns + 2*np.pi) % (2*np.pi)
            # vector from start (x_prev,y_prev)
            dx = lengths * np.sin(headings)
            dy = lengths * np.cos(headings)
            x_end = r["x_prev"] + dx
            y_end = r["y_prev"] + dy
            for k in range(self.K):
                rows.append({
                    "stratum_id": idx,
                    "id": r["id"],
                    "time": r["time"],
                    "x_end": float(x_end[k]),
                    "y_end": float(y_end[k]),
                    "heading_step": float(headings[k]),
                    "is_used": 0,
                })
        return pd.DataFrame(rows)

def build_used_available(reg: pd.DataFrame, cfg) -> pd.DataFrame:
    steps = reg[[cfg.id_col, cfg.time_col, "x_m","y_m","x_prev","y_prev",
                 "heading_rad","heading_prev_rad","step_len_m","turn_rad"]].copy()
    steps = steps.rename(columns={cfg.id_col:"id", cfg.time_col:"time"})
    steps["stratum_id"] = steps.index

    used = pd.DataFrame({
        "stratum_id": steps["stratum_id"],
        "id": steps["id"],
        "time": steps["time"],
        "x_end": steps["x_m"],
        "y_end": steps["y_m"],
        "heading_step": steps["heading_rad"],
        "log_l": np.log(steps["step_len_m"] + 1e-9),
        "cos_turn": np.cos(steps["turn_rad"]),
        "is_used": 1,
    })
    if cfg.include_log_l2:
        used["log_l2"] = used["log_l"]**2

    sampler = AvailableSampler(K=cfg.K_available, seed=cfg.seed)
    avail = sampler.sample(steps)

    # recompute log_l for available
    dx = avail["x_end"] - steps.loc[avail["stratum_id"], "x_prev"].to_numpy()
    dy = avail["y_end"] - steps.loc[avail["stratum_id"], "y_prev"].to_numpy()
    avail["log_l"] = np.log(np.hypot(dx, dy) + 1e-9)
    avail["cos_turn"] = 1.0  # placeholder; can add proper turn later
    if cfg.include_log_l2:
        avail["log_l2"] = avail["log_l"]**2

    return pd.concat([used, avail], ignore_index=True)
