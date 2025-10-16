from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    path: str
    id_col: str = "Tag_shortName"
    time_col: str = "DateTimeUTC"
    lon_col: str = "Longitude"
    lat_col: str = "Latitude"
    sex_col: Optional[str] = "sex"

    dt_minutes: int = 10
    K_available: int = 20

    neighbor_radius_m: float = 500.0
    cone_half_angle_deg: float = 60.0

    include_cos_turn: bool = True
    include_log_l: bool = True
    include_log_l2: bool = True

    seed: int = 42
