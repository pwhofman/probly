"""Mock runs for the Milton demo: a made-up forecast ensemble around Hurricane Milton's real track.

The runs are a simple wind model of a storm moved along the observed track (IBTrACS), with offsets
per model (the part that the models disagree on) and per run (the part that changes from run to run
of the same model), so that the demo's views have something plausible to show where real runs are
missing. ``pack.py`` uses them for the full WeatherNext models until those have run; run alone, this
script writes ``../assets/data.js`` with mock runs only. Run from the repository's root (it downloads
IBTrACS and Natural Earth into ``cache/`` once)::

    uv run python docs/source/demos/milton/compute/mock.py
"""

from __future__ import annotations

from common import TIMES, grid, land, observed_track, on_land, points, summarize, write
import numpy as np

# The made-up models: offsets of each model's runs from the observed track at 72 h (km along and
# to the left of the storm's motion), its intensity relative to the gridded observed intensity,
# and its bias of the trade winds (m/s). The runs of one model scatter around these offsets.
MODELS = {
    "full": {
        "name": "WeatherNext Cyclones",
        "label": "Full",
        "detail": "0.25°, shown at 0.5°",
        "step": 0.5,
        "rmax": 55.0,
        "intensity": 0.6,
        "along": [-45.0, 30.0, -10.0, 55.0],
        "left": [60.0, 20.0, 75.0, 5.0],
        "strength": [0.94, 1.05, 0.97, 1.08],
        "trades": [-1.4, 0.6, 1.5, -0.7],
        "runs": 8,
        "window": 24,
        "run_along": 70.0,
        "run_left": 55.0,
        "run_strength": 0.08,
    },
    "mini": {
        "name": "WeatherNext Cyclones Mini",
        "label": "Mini",
        "detail": "1°",
        "step": 1.0,
        "rmax": 110.0,
        "intensity": 0.42,
        "along": [20.0],
        "left": [45.0],
        "strength": [1.0],
        "trades": [0.0],
        "runs": 8,
        "window": 14,
        "run_along": 95.0,
        "run_left": 75.0,
        "run_strength": 0.1,
    },
}


def interpolate_track(track: list[tuple[float, float, float, float]], hours: np.ndarray) -> np.ndarray:
    """The observed track at the given hours: lat, lon and maximum wind."""
    table = np.array(track)
    return np.stack([np.interp(hours, table[:, 0], table[:, k]) for k in (1, 2, 3)], axis=-1)


def displace(base: np.ndarray, hours: np.ndarray, along: np.ndarray, left: np.ndarray) -> np.ndarray:
    """Move track positions by km along and to the left of the storm's motion."""
    lat, lon = base[:, 0], base[:, 1]
    ahead = np.gradient(np.column_stack([lon * np.cos(np.radians(lat)), lat]), hours, axis=0)
    ahead /= np.linalg.norm(ahead, axis=1, keepdims=True)
    normal = np.column_stack([-ahead[:, 1], ahead[:, 0]])
    shift = (along[:, None] * ahead + left[:, None] * normal) / 111.2  # in degrees of latitude
    moved = base.copy()
    moved[:, 0] = lat + shift[:, 1]
    moved[:, 1] = lon + shift[:, 0] / np.cos(np.radians(lat))
    return moved


def background(lat: np.ndarray, lon: np.ndarray, hours: float, rng: np.random.Generator) -> np.ndarray:
    """A smooth wind field without the storm: trade winds, westerlies, and slow eddies (u, v)."""
    u = -6.5 + 0.55 * (lat - 18.0) + 1.5 * np.sin(np.radians(lon * 9 + hours * 2))
    v = 1.2 * np.cos(np.radians(lat * 14 - lon * 6 + hours * 3)) + 0.8 * np.sin(np.radians(lon * 11))
    u += rng.normal(0, 0.5) * np.sin(np.radians(lat * 20 + lon * 13))
    v += rng.normal(0, 0.5) * np.cos(np.radians(lat * 17 - lon * 9))
    return np.stack([u, v])


def storm(lat: np.ndarray, lon: np.ndarray, center: np.ndarray, motion: np.ndarray, rmax: float) -> np.ndarray:
    """The wind of a cyclone (u, v): a vortex turning counterclockwise, plus a share of its motion."""
    clat, clon, vmax = center
    dx = (lon - clon) * 111.2 * np.cos(np.radians(clat))
    dy = (lat - clat) * 111.2
    r = np.hypot(dx, dy) + 1e-6
    speed = np.where(r < rmax, vmax * r / rmax, vmax * (rmax / r) ** 0.55)
    fade = np.exp(-r / 500)
    return np.stack([-speed * dy / r + 0.5 * motion[0] * fade, speed * dx / r + 0.5 * motion[1] * fade])


def wind(
    lat: np.ndarray,
    lon: np.ndarray,
    land_mask: np.ndarray,
    track: np.ndarray,
    index: int,
    rmax: float,
    trades: float,
    eddies: tuple[float, float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """The wind speed of one run at one lead time (m/s).

    ``eddies`` (amplitude, two phases) are weather systems of the run's own, which grow with the lead
    time: small and different for every run, larger for what happened, which no run foresaw.
    """
    hours = float(TIMES[index])
    previous, following = track[max(index - 1, 0)], track[min(index + 1, len(track) - 1)]
    span = 6.0 * (min(index + 1, len(track) - 1) - max(index - 1, 0))
    motion = np.array(
        [
            (following[1] - previous[1]) * 111.2 * 1000 * np.cos(np.radians(track[index, 0])) / (span * 3600),
            (following[0] - previous[0]) * 111.2 * 1000 / (span * 3600),
        ]
    )
    u, v = background(lat, lon, hours, rng) + storm(lat, lon, track[index], motion, rmax)
    u += trades * np.exp(-np.abs(lat - 22) / 6)
    amplitude, phase_u, phase_v = eddies
    grow = amplitude * (0.25 + hours / 96)
    u += grow * np.sin(np.radians(lat * 23 + lon * 17 + phase_u + hours * 4))
    v += grow * np.cos(np.radians(lat * 19 - lon * 21 + phase_v + hours * 5))
    return np.hypot(u, v) * np.where(land_mask, 0.72, 1.0)


def build_model(key: str, rings: list[np.ndarray], observed: list) -> dict:
    """The runs of one made-up model, summarized as the demo needs them."""
    spec = MODELS[key]
    cells = grid(spec["step"])
    lat, lon = points(cells)
    land_mask = on_land(rings, lat, lon)
    hours = TIMES.astype(float)
    base = interpolate_track(observed, hours)
    base[:, 2] *= spec["intensity"]
    growth = TIMES / 72.0
    rng = np.random.default_rng(7 if key == "full" else 11)

    tracks, winds = [], []
    for along, left, strength, trades in zip(
        spec["along"], spec["left"], spec["strength"], spec["trades"], strict=True
    ):
        model_tracks, model_winds = [], []
        for _ in range(spec["runs"]):
            run_along = rng.normal(0, spec["run_along"]) * growth + rng.normal(0, 15) * growth**2
            run_left = rng.normal(0, spec["run_left"]) * growth + rng.normal(0, 12) * growth**2
            track = displace(base, hours, along * growth**1.3 + run_along, left * growth**1.3 + run_left)
            track[:, 2] *= strength * (1 + rng.normal(0, spec["run_strength"]) * np.sqrt(growth))
            rmax = spec["rmax"] * (1 + rng.normal(0, 0.12))
            eddies = (0.5, rng.uniform(0, 360), rng.uniform(0, 360))
            model_tracks.append(track)
            model_winds.append(
                [wind(lat, lon, land_mask, track, t, rmax, trades, eddies, rng) for t in range(len(TIMES))]
            )
        tracks.append(model_tracks)
        winds.append(model_winds)

    # what happened: the same wind model along the observed track, on the same grid
    eddies = (2.0, 40.0, 210.0)
    truth = np.array(
        [
            wind(lat, lon, land_mask, base, t, spec["rmax"] * 0.9, 0.0, eddies, np.random.default_rng(99))
            for t in range(len(TIMES))
        ]
    )
    summary = summarize(np.array(winds), truth, np.array(tracks), rings, cells, spec["window"])
    return {
        "name": spec["name"],
        "label": spec["label"],
        "detail": spec["detail"],
        "mock": True,
        "group": ["model", "models"],
        "about": "Mock data: the runs are invented around Milton's real track, to try out the design.",
        **summary,
    }


def main() -> None:
    """Write the demo's data with mock runs only."""
    observed = observed_track()
    rings = land()
    models = {key: build_model(key, rings, observed) for key in MODELS}
    write(
        models,
        observed,
        rings,
        "The data of the Milton demo, written by compute/mock.py. MOCK DATA: every forecast in it is\n"
        "invented; only the observed track (IBTrACS) and the coastlines (Natural Earth) are real.",
    )


if __name__ == "__main__":
    main()
