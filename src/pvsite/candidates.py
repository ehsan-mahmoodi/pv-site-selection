"""Candidate site generation.

The brief this models: a pre-study has screened a region and produced roughly 500
candidate points. Each one carries the attributes a developer would have collected
before committing to anything expensive - irradiation, terrain, land cost, and how
far it sits from the grid and the road network.

Attributes are correlated the way real geography is. Irradiation rises to the south
and with altitude; land is cheap where it is far from a road; substations cluster
near population, which is also where land costs most. Those correlations are the
whole reason the problem is multi-objective - if the best-irradiated site were also
the cheapest to connect, there would be nothing to decide.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RegionSpec:
    """Bounding box and physical character of the screening region."""

    name: str = "Yazd province"
    lat_min: float = 30.6
    lat_max: float = 33.4
    lon_min: float = 52.8
    lon_max: float = 56.4
    ghi_south_kwh: float = 2150.0   # annual GHI at the southern edge
    ghi_north_kwh: float = 1870.0   # and at the northern edge
    n_substations: int = 9
    n_roads: int = 6


def generate(
    region: RegionSpec | None = None, n: int = 500, seed: int = 20260921
) -> pd.DataFrame:
    """Produce `n` candidate sites with correlated, physically plausible attributes."""
    region = region or RegionSpec()
    rng = np.random.default_rng(seed)

    lat = rng.uniform(region.lat_min, region.lat_max, n)
    lon = rng.uniform(region.lon_min, region.lon_max, n)

    # Irradiation: latitude gradient, altitude bonus, local variation.
    span = region.lat_max - region.lat_min
    south_frac = (region.lat_max - lat) / span
    altitude = 900.0 + 1400.0 * rng.beta(2.0, 3.0, n)
    ghi = (
        region.ghi_north_kwh
        + (region.ghi_south_kwh - region.ghi_north_kwh) * south_frac
        + 0.035 * (altitude - 1200.0)
        + rng.normal(0.0, 28.0, n)
    )

    # Substations and roads, placed then measured against.
    sub_lat = rng.uniform(region.lat_min, region.lat_max, region.n_substations)
    sub_lon = rng.uniform(region.lon_min, region.lon_max, region.n_substations)
    sub_capacity = rng.choice([20.0, 30.0, 50.0, 80.0], region.n_substations)

    road_lat = rng.uniform(region.lat_min, region.lat_max, region.n_roads)
    road_lon = rng.uniform(region.lon_min, region.lon_max, region.n_roads)

    grid_km, nearest_sub = _nearest(lat, lon, sub_lat, sub_lon)
    road_km, _ = _nearest(lat, lon, road_lat, road_lon)

    # Terrain: steeper ground costs more to prepare and caps usable area.
    slope = np.clip(rng.gamma(2.0, 1.6, n), 0.0, 18.0)
    area_ha = np.clip(rng.gamma(4.0, 26.0, n) * (1.0 - slope / 40.0), 20.0, 400.0)

    # Land price: falls with distance from road, rises near substations
    # (which sit near towns), plus noise.
    land_eur_ha = (
        11000.0
        - 130.0 * np.clip(road_km, 0, 60)
        + 95.0 * np.clip(60 - grid_km, 0, 60)
        + rng.normal(0.0, 1400.0, n)
    )
    land_eur_ha = np.clip(land_eur_ha, 1800.0, 26000.0)

    # 1.6 ha per MWp is a reasonable fixed-tilt footprint.
    capacity_mw = np.clip(np.floor(area_ha / 1.6), 5.0, 120.0)

    df = pd.DataFrame(
        {
            "site_id": [f"S{i:03d}" for i in range(n)],
            "lat": np.round(lat, 4),
            "lon": np.round(lon, 4),
            "altitude_m": np.round(altitude, 0),
            "ghi_kwh_m2_yr": np.round(ghi, 1),
            "slope_deg": np.round(slope, 2),
            "area_ha": np.round(area_ha, 1),
            "capacity_mw": capacity_mw,
            "grid_km": np.round(grid_km, 2),
            "road_km": np.round(road_km, 2),
            "land_eur_ha": np.round(land_eur_ha, 0),
            "substation": nearest_sub,
        }
    )
    df["substation_capacity_mw"] = sub_capacity[nearest_sub]
    return df


def _nearest(
    lat: np.ndarray, lon: np.ndarray, t_lat: np.ndarray, t_lon: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Great-circle distance in km to the nearest target, and which one."""
    d = np.stack([haversine_km(lat, lon, a, b) for a, b in zip(t_lat, t_lon)], axis=1)
    return d.min(axis=1), d.argmin(axis=1)


def haversine_km(
    lat1: np.ndarray, lon1: np.ndarray, lat2: float, lon2: float
) -> np.ndarray:
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
