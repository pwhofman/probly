r"""The data of the Milton demo from WeatherNext's runs: ``../assets/data.js``.

Reads the runs that ``forecast.py`` wrote (``<results>/<checkpoint>/member_<seed>.nc``) and their
initial fields (``<results>/inputs_t0.nc``), and summarizes them as the demo needs them
(``common.summarize``): the wind on the demo's grid and at its places every 6 hours, every run's
track, found as the lowest sea-level pressure near the run's last position, and the split of the
runs' spread into aleatoric and epistemic variance, with probly. Models whose runs are missing are
mock data (``mock_runs.py``); for now, these are the full models, which need a larger GPU than we have
run them on.

There is no "what happened" for the wind yet: the targets in WeatherNext's public data are ECMWF's
HRES forecast from the same start, not observations. Milton's observed track (IBTrACS) is.

Run from the repository's root, with xarray to read the runs (not one of probly's dependencies)::

    uv run --with xarray --with h5netcdf --with h5py python docs/source/demos/milton/compute/pack.py \
        --results path/to/results
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common import HERE, TIMES, grid, land, observed_track, points, summarize, write
from mock_runs import build_model
import numpy as np
import xarray as xr

# The models of the demo whose runs exist, and the checkpoints whose runs stand in for its models.
MODELS = {
    "mini": {
        "name": "WeatherNext Cyclones Mini",
        "label": "Mini",
        "detail": "1°",
        "step": 1.0,
        "window": 14,
        # trained on data through 2023 and through 2022, both before Milton
        "checkpoints": ["mini_2024", "mini_2023"],
        "group": ["checkpoint", "checkpoints"],
        "about": (
            "Real runs, computed on a CPU: 8 of each of the two public Mini checkpoints, trained on data "
            "through 2023 and through 2022, before Milton. The two checkpoints stand in for WeatherNext's "
            "four models, so their split is rough. What happened is Milton's observed track; the observed "
            "wind is still to come."
        ),
    },
}
WIND = ("10m_u_component_of_wind", "10m_v_component_of_wind")
PRESSURE = "mean_sea_level_pressure"
SEARCH = 3.0  # degrees around the last position where a run's center may be next
CORE = 2.5  # degrees around the center where its maximum wind is


def load(path: Path, initial: xr.Dataset) -> xr.Dataset:
    """A run's fields at the demo's lead times (hours), the initial fields first."""
    run = xr.open_dataset(path, engine="h5netcdf")
    hours = (run.time.values / np.timedelta64(1, "h")).astype(int)
    run = run.assign_coords(time=hours).sel(time=[h for h in TIMES if h > 0]).drop_vars("datetime", errors="ignore")
    first = initial[list(run.data_vars)].expand_dims(time=[0])
    return xr.concat([first, run], dim="time").load()


def bilinear(field: np.ndarray, lats: np.ndarray, lons: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """A field on a regular grid (..., lat, lon; both ascending) at points, interpolated bilinearly."""
    i = np.clip((lat - lats[0]) / (lats[1] - lats[0]), 0, len(lats) - 1.000001)
    j = np.clip((lon - lons[0]) / (lons[1] - lons[0]), 0, len(lons) - 1.000001)
    i0, j0 = np.floor(i).astype(int), np.floor(j).astype(int)
    wi, wj = i - i0, j - j0
    return (
        field[..., i0, j0] * (1 - wi) * (1 - wj)
        + field[..., i0 + 1, j0] * wi * (1 - wj)
        + field[..., i0, j0 + 1] * (1 - wi) * wj
        + field[..., i0 + 1, j0 + 1] * wi * wj
    )


def vertex(left: float, middle: float, right: float) -> float:
    """Where a parabola through three equally spaced values has its extremum, from -0.5 to 0.5."""
    curvature = left - 2 * middle + right
    return float(np.clip(0.5 * (left - right) / curvature, -0.5, 0.5)) if curvature > 0 else 0.0


def track(run: xr.Dataset, start: tuple[float, float]) -> np.ndarray:
    """A run's track: per lead time the lowest pressure near its last position, and its maximum wind.

    The lowest pressure is refined between grid points by a parabola in each direction. Returns the
    latitude, the longitude (east of Greenwich negative) and the maximum wind within ``CORE`` degrees.
    """
    lats, lons = run.lat.values, run.lon.values
    pressure = run[PRESSURE].values
    speed = np.hypot(run[WIND[0]].values, run[WIND[1]].values)
    lat, lon = start[0], start[1] % 360
    out = []
    for t in range(len(TIMES)):
        near = (np.abs(lats[:, None] - lat) <= SEARCH) & (np.abs(lons[None, :] - lon) <= SEARCH)
        i, j = np.unravel_index(np.argmin(np.where(near, pressure[t], np.inf)), near.shape)
        di = vertex(*pressure[t, i - 1 : i + 2, j]) if 0 < i < len(lats) - 1 else 0.0
        dj = vertex(*pressure[t, i, j - 1 : j + 2]) if 0 < j < len(lons) - 1 else 0.0
        lat = lats[i] + di * (lats[1] - lats[0])
        lon = lons[j] + dj * (lons[1] - lons[0])
        distance = np.hypot(lats[:, None] - lat, (lons[None, :] - lon) * np.cos(np.radians(lat)))
        out.append([lat, lon - 360 if lon > 180 else lon, float(speed[t][distance <= CORE].max())])
    return np.array(out)


def build_real(key: str, results: Path, rings: list[np.ndarray], observed: list) -> dict | None:
    """The runs of a model from its checkpoints' files, or ``None`` if they are missing."""
    spec = MODELS[key]
    folders = [results / checkpoint for checkpoint in spec["checkpoints"]]
    files = [sorted(folder.glob("member_*.nc")) for folder in folders]
    if not all(files) or len({len(group) for group in files}) > 1:
        return None
    initial = xr.open_dataset(results / "inputs_t0.nc", engine="h5netcdf").isel(time=0, drop=True)
    initial = initial.drop_vars("datetime", errors="ignore").load()
    cells = grid(spec["step"])
    lat, lon = points(cells)
    table = np.array(observed)
    start = (float(np.interp(0, table[:, 0], table[:, 1])), float(np.interp(0, table[:, 0], table[:, 2])))
    winds, tracks = [], []
    for group in files:
        group_winds, group_tracks = [], []
        for path in group:
            run = load(path, initial)
            speed = np.hypot(run[WIND[0]].values, run[WIND[1]].values)
            group_winds.append(bilinear(speed, run.lat.values, run.lon.values, lat, lon % 360))
            group_tracks.append(track(run, start))
        winds.append(group_winds)
        tracks.append(group_tracks)
    summary = summarize(np.array(winds), None, np.array(tracks), rings, cells, spec["window"])
    return {
        "name": spec["name"],
        "label": spec["label"],
        "detail": spec["detail"],
        "mock": False,
        "group": spec["group"],
        "about": spec["about"],
        **summary,
    }


def main() -> None:
    """Write the demo's data: the real runs where they exist, mock runs elsewhere."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, default=HERE / "results", help="the folder forecast.py wrote")
    args = parser.parse_args()
    observed = observed_track()
    rings = land()
    models = {"full": build_model("full", rings, observed)}
    for key in MODELS:
        models[key] = build_real(key, args.results, rings, observed) or build_model(key, rings, observed)
    real = ", ".join(model["name"] for model in models.values() if not model["mock"]) or "none"
    write(
        models,
        observed,
        rings,
        "The data of the Milton demo, written by compute/pack.py.\n"
        f"Real runs: {real}. Every other model is MOCK DATA (compute/mock_runs.py).",
    )


if __name__ == "__main__":
    main()
