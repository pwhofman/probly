r"""The data of the Milton demo from WeatherNext's runs: ``../assets/data.js``.

Reads the runs that ``forecast.py`` wrote (``<results>/<model>/<checkpoint>/member_<seed>.nc``, with
the initial fields in ``<results>/<model>/inputs_t0.nc``; the demo's model is ``full``, WeatherNext
Cyclones), and summarizes them as the demo needs them (``common.summarize``): the wind on the demo's
grid and at its places every 6 hours, every run's track, found as the lowest sea-level pressure near
the run's last position, and the split of the runs' uncertainty into aleatoric and epistemic
variance, with probly. A model whose runs are missing keeps its real runs from the current
``data.js``, if it has them, and else is mock data (``mock_runs.py``). ``run_full.py`` calls this
script after its runs.

There is no "what happened" for the wind yet: the targets in WeatherNext's public data are ECMWF's
HRES forecast from the same start, not observations. Milton's observed track (IBTrACS) is.

Run from the repository's root, in WeatherNext's environment or in probly's with xarray to read the
runs (not one of probly's dependencies)::

    uv run --with xarray --with h5netcdf --with h5py python docs/source/demos/milton/compute/pack.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import HERE, OUT, TIMES, grid, land, observed_track, points, summarize, write
from mock_runs import build_model
import numpy as np
import xarray as xr

# The model of the demo, and the checkpoints whose runs it is (the code takes several models).
MODELS = {
    "full": {
        "name": "WeatherNext Cyclones",
        "detail": "0.25°, shown at 0.5°",
        "step": 0.5,
        "window": 24,
        # WeatherNext's four models, trained on data through 2023, before Milton
        "checkpoints": ["model1", "model2", "model3", "model4"],
        "group": ["model", "models"],
        "about": "8 runs of each of WeatherNext's four models, trained on data through 2023, before Milton.",
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
    latitude, the longitude (west of Greenwich negative) and the maximum wind within ``CORE`` degrees.
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
    folders = [results / key / checkpoint for checkpoint in spec["checkpoints"]]
    # the runs in the order of their seeds (member_<seed>.nc)
    files = [sorted(folder.glob("member_*.nc"), key=lambda path: int(path.stem.split("_")[1])) for folder in folders]
    if not all(files) or len({len(group) for group in files}) > 1:
        counts = ", ".join(f"{folder.name}: {len(group)}" for folder, group in zip(folders, files, strict=True))
        print(f"{spec['name']}: no complete runs in {results / key} ({counts})")  # noqa: T201
        return None
    print(f"{spec['name']}: {sum(map(len, files))} runs in {results / key}")  # noqa: T201
    initial = xr.open_dataset(results / key / "inputs_t0.nc", engine="h5netcdf").isel(time=0, drop=True)
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
        "detail": spec["detail"],
        "mock": False,
        "group": spec["group"],
        "about": spec["about"],
        **summary,
    }


def existing() -> dict:
    """The models of the current ``../assets/data.js`` whose runs are real."""
    if not OUT.exists():
        return {}
    text = OUT.read_text(encoding="utf-8")
    data = json.loads(text.split("window.PROBLY_DEMO_MILTON = ", 1)[1].strip().removesuffix(";"))
    return {key: model for key, model in data["models"].items() if not model.get("mock")}


def pack(results: Path, require: tuple[str, ...] = ()) -> None:
    """Write the demo's data: per model its runs in ``results``, else its real runs so far, else mock runs.

    Raises:
        RuntimeError: If a model in ``require`` has no complete runs in ``results``.
    """
    observed = observed_track()
    rings = land()
    kept = existing()
    models = {}
    for key in MODELS:
        model = build_real(key, results, rings, observed)
        if model is None and key in require:
            msg = f"{MODELS[key]['name']}: its runs in {results / key} are missing or incomplete"
            raise RuntimeError(msg)
        if model is None and key in kept:
            print(f"{MODELS[key]['name']}: its runs so far, from {OUT.name}")  # noqa: T201
            model = kept[key]
        if model is None:
            print(f"{MODELS[key]['name']}: MOCK runs, made up by mock_runs.py")  # noqa: T201
            model = build_model(key, rings, observed)
        models[key] = model
    real = ", ".join(model["name"] for model in models.values() if not model["mock"]) or "none"
    write(
        models,
        observed,
        rings,
        f"The data of the Milton demo, written by compute/pack.py. Runs from WeatherNext: {real}.",
    )


def main() -> None:
    """Write the demo's data from the runs."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, default=HERE / "results", help="the folder of the runs")
    pack(parser.parse_args().results)


if __name__ == "__main__":
    main()
