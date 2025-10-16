import numpy as np
import pandas as pd
import math
try:
    from scipy.spatial import cKDTree as KDTree
except Exception:
    KDTree = None

class TimeNeighborIndex:
    def __init__(self, radius_m: float, cone_half_angle_deg: float):
        self.radius_m = radius_m
        self.cone_half_angle_deg = cone_half_angle_deg
        self.time_to_tree = {}
        self.time_to_arr = {}

    def build(self, reg: pd.DataFrame, id_col: str, time_col: str):
        for t, g in reg.groupby(time_col):
            xy = np.c_[g["x_m"].to_numpy(), g["y_m"].to_numpy()]
            self.time_to_tree[t] = KDTree(xy) if KDTree is not None and len(xy)>1 else None
            self.time_to_arr[t] = {
                "x": g["x_m"].to_numpy(),
                "y": g["y_m"].to_numpy(),
                "id": g[id_col].to_numpy(),
                "heading": g["heading_rad"].to_numpy(),
                "speed": g["speed_mps"].to_numpy(),
            }
        return self

    def query_social(self, time, x_end: float, y_end: float, step_heading: float, focal_id):
        arr = self.time_to_arr.get(time)
        if arr is None:
            return {"nn_dist": np.nan, "n_forward": 0.0, "n_behind": 0.0,
                    "ahead_any": 0.0, "behind_any": 0.0,
                    "mean_align_fwd": np.nan, "rel_speed_fwd": np.nan}
        dx = arr["x"] - x_end; dy = arr["y"] - y_end
        dist = np.hypot(dx, dy)
        mask = (arr["id"] != focal_id) & np.isfinite(dist)
        dx = dx[mask]; dy = dy[mask]; dist = dist[mask]
        heading_nb = arr["heading"][mask]; speed_nb = arr["speed"][mask]

        in_r = dist <= self.radius_m
        if not np.any(in_r):
            return {"nn_dist": np.nan, "n_forward": 0.0, "n_behind": 0.0,
                    "ahead_any": 0.0, "behind_any": 0.0,
                    "mean_align_fwd": np.nan, "rel_speed_fwd": np.nan}

        dx = dx[in_r]; dy = dy[in_r]; dist = dist[in_r]
        heading_nb = heading_nb[in_r]; speed_nb = speed_nb[in_r]
        # step unit vector
        ux = math.sin(step_heading); uy = math.cos(step_heading)
        proj = (dx*ux + dy*uy)
        half = math.radians(self.cone_half_angle_deg)
        ang_to_nb = np.arctan2(dx, dy)
        ang_diff = np.abs(((ang_to_nb - step_heading + np.pi) % (2*np.pi)) - np.pi)
        forward = (proj > 0) & (ang_diff <= half)
        behind  = (proj < 0) & (ang_diff <= half)

        n_fwd = float(np.sum(forward)); n_bhd = float(np.sum(behind))
        ahead_any = 1.0 if n_fwd > 0 else 0.0
        behind_any = 1.0 if n_bhd > 0 else 0.0

        mean_align_fwd = float(np.nan)
        rel_speed_fwd = float(np.nan)
        if n_fwd > 0:
            align = np.cos(((heading_nb[forward] - step_heading + np.pi) % (2*np.pi)) - np.pi)
            mean_align_fwd = float(np.nanmean(align))
            rel_speed_fwd = float(np.nanmean(speed_nb[forward] - 0.0))
        nn = float(np.min(dist)) if dist.size else float("nan")
        return {
            "nn_dist": nn,
            "n_forward": n_fwd,
            "n_behind": n_bhd,
            "ahead_any": ahead_any,
            "behind_any": behind_any,
            "mean_align_fwd": mean_align_fwd,
            "rel_speed_fwd": rel_speed_fwd,
        }
