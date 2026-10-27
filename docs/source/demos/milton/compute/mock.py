"""Mock data for the Milton demo: a made-up forecast ensemble around Hurricane Milton's real track.

Writes ``../assets/data.js`` in the format that the real pipeline will write from WeatherNext
forecasts, so that the demo can be designed and reviewed before the forecasts exist. Only the
observed track (IBTrACS) and the coastlines (Natural Earth) are real. Every forecast in the file
is invented: the runs are a simple wind model of a storm moved along the observed track, with
offsets per model (the part that the models disagree on) and per run (the part that changes from
run to run of the same model), so that the demo's views have something plausible to show.

Run from this folder (it downloads IBTrACS and Natural Earth into ``cache/`` once)::

    python mock.py
"""

from __future__ import annotations

import base64
import csv
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
import urllib.request

from matplotlib.path import Path as Polygon
import numpy as np

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
OUT = HERE.parent / "assets" / "data.js"

IBTRACS_URL = (
    "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/"
    "v04r01/access/csv/ibtracs.NA.list.v04r01.csv"
)
LAND_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_land.geojson"
MILTON = "2024279N21265"  # Milton's IBTrACS storm id

INIT = datetime(2024, 10, 7, tzinfo=UTC)  # the initialization of the public WeatherNext data
TIMES = np.arange(0, 97, 6)  # lead times in hours, every 6 h for four days
LON = (-97.0, -72.0)  # the map: the Gulf of Mexico, Florida and the western Atlantic
LAT = (18.0, 33.0)
KNOT = 0.514444  # m/s
GALE = 34 * KNOT  # tropical-storm-force wind, NHC's lowest wind speed probability threshold

PLACES = [  # the places the demo offers, west to east
    ("Mérida", 20.97, -89.62),
    ("Havana", 23.11, -82.37),
    ("Tampa", 27.95, -82.46),
    ("Sarasota", 27.34, -82.53),
    ("Fort Myers", 26.64, -81.87),
    ("Orlando", 28.54, -81.38),
]

COAST = [  # places on Florida's Gulf coast, south to north, to name where a run comes ashore
    ("Naples", 26.14, -81.79),
    ("Fort Myers Beach", 26.45, -81.95),
    ("Boca Grande", 26.75, -82.26),
    ("Venice", 27.10, -82.45),
    ("Siesta Key", 27.27, -82.55),
    ("Longboat Key", 27.41, -82.66),
    ("Anna Maria Island", 27.53, -82.73),
    ("St. Petersburg", 27.77, -82.68),
    ("Clearwater", 27.97, -82.80),
    ("Tarpon Springs", 28.15, -82.76),
    ("Hernando Beach", 28.48, -82.66),
    ("Crystal River", 28.90, -82.63),
    ("Cedar Key", 29.14, -83.03),
]

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


def download(url: str, name: str) -> Path:
    """Download a file into the cache once."""
    path = CACHE / name
    if not path.exists():
        CACHE.mkdir(exist_ok=True)
        print(f"downloading {url}")  # noqa: T201
        urllib.request.urlretrieve(url, path)  # noqa: S310
    return path


def observed_track() -> list[tuple[float, float, float, float]]:
    """Milton's best track (hours since INIT, lat, lon, maximum wind in m/s), from IBTrACS."""
    rows = []
    with download(IBTRACS_URL, "ibtracs_na.csv").open(newline="") as file:
        for row in csv.DictReader(file):
            if row["SID"] != MILTON:
                continue
            time = datetime.fromisoformat(row["ISO_TIME"]).replace(tzinfo=UTC)
            hours = (time - INIT) / timedelta(hours=1)
            if -6 <= hours <= 102:
                rows.append((hours, float(row["LAT"]), float(row["LON"]), float(row["USA_WIND"]) * KNOT))
    return rows


def clip(ring: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    """Clip a polygon to a rectangle (Sutherland-Hodgman)."""
    west, east, south, north = box
    edges = [(west, 0, 1), (east, 0, -1), (south, 1, 1), (north, 1, -1)]  # edge, axis, inward side
    points = ring
    for edge, axis, side in edges:
        if len(points) == 0:
            break
        kept = []
        flags = side * (points[:, axis] - edge) >= 0
        for i in range(len(points)):
            a, b = points[i - 1], points[i]
            ina, inb = flags[i - 1], flags[i]
            if ina != inb:
                share = (edge - a[axis]) / (b[axis] - a[axis])
                kept.append(a + share * (b - a))
            if inb:
                kept.append(b)
        points = np.array(kept)
    return points


def simplify(ring: np.ndarray, tolerance: float) -> np.ndarray:
    """Simplify a line (Douglas-Peucker)."""
    if len(ring) < 3:
        return ring
    start, end = ring[0], ring[-1]
    direction = end - start
    length = np.hypot(*direction)
    if length == 0:
        distances = np.hypot(*(ring - start).T)
    else:
        offset = ring - start
        distances = np.abs(direction[0] * offset[:, 1] - direction[1] * offset[:, 0]) / length
    index = int(np.argmax(distances))
    if distances[index] <= tolerance:
        return np.array([start, end])
    left = simplify(ring[: index + 1], tolerance)
    right = simplify(ring[index:], tolerance)
    return np.vstack([left[:-1], right])


def land() -> list[np.ndarray]:
    """The land of the map, as polygons clipped to the map (with a margin) and simplified."""
    features = json.loads(download(LAND_URL, "ne_50m_land.geojson").read_text())["features"]
    box = (LON[0] - 1, LON[1] + 1, LAT[0] - 1, LAT[1] + 1)
    rings = []
    for feature in features:
        geometry = feature["geometry"]
        polygons = geometry["coordinates"] if geometry["type"] == "MultiPolygon" else [geometry["coordinates"]]
        for polygon in polygons:
            ring = np.array(polygon[0], dtype=float)
            if ring[:, 0].max() < box[0] or ring[:, 0].min() > box[1]:
                continue
            if ring[:, 1].max() < box[2] or ring[:, 1].min() > box[3]:
                continue
            clipped = clip(ring, box)
            if len(clipped) >= 3:
                rings.append(simplify(clipped, 0.02))
    return [ring for ring in rings if len(ring) >= 3]


def on_land(rings: list[np.ndarray], lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Whether points are on land."""
    points = np.column_stack([lon.ravel(), lat.ravel()])
    inside = np.zeros(len(points), dtype=bool)
    for ring in rings:
        inside |= Polygon(ring).contains_points(points)
    return inside.reshape(lat.shape)


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


def landfall(rings: list[np.ndarray], track: np.ndarray) -> dict | None:
    """Where and when a track first comes ashore in Florida (half-hourly)."""
    hours = np.arange(TIMES[0], TIMES[-1] + 0.25, 0.5)
    lat = np.interp(hours, TIMES, track[:, 0])
    lon = np.interp(hours, TIMES, track[:, 1])
    ashore = on_land(rings, lat, lon) & (lon > -88) & (lat > 24.6)
    if not ashore.any():
        return None
    first = int(np.argmax(ashore))
    return {"hours": float(hours[first]), "lat": round(float(lat[first]), 2), "lon": round(float(lon[first]), 2)}


def encode(values: np.ndarray, scale: float) -> str:
    """Values on the grid as bytes (value * scale, rounded and clipped), in base64."""
    return base64.b64encode(np.clip(np.round(values * scale), 0, 255).astype(np.uint8).tobytes()).decode()


def windows(winds: np.ndarray, tracks: list[np.ndarray], step: float, size: int) -> dict:
    """Every run's wind near the storm, for the contours of every run.

    A square of ``size`` x ``size`` cells per lead time that follows the runs' mean position; ``winds``
    has the shape (runs, times, rows, cols).
    """
    rows, cols = winds.shape[2:]
    origins, fields = [], []
    for t in range(len(TIMES)):
        center = np.mean([track[t] for track in tracks], axis=0)
        row = int(np.clip(round((LAT[1] - center[0]) / step) - size // 2, 0, rows - size))
        col = int(np.clip(round((center[1] - LON[0]) / step) - size // 2, 0, cols - size))
        origins.append([row, col])
        fields.append(encode(winds[:, t, row : row + size, col : col + size], 4))
    return {"size": size, "origins": origins, "fields": fields}


def build_model(key: str, rings: list[np.ndarray], observed: list) -> dict:
    """The runs of one made-up model, summarized as the demo needs them."""
    spec = MODELS[key]
    step = spec["step"]
    lats = np.arange(LAT[1], LAT[0] - step / 2, -step)  # north to south: rows of the image
    lons = np.arange(LON[0], LON[1] + step / 2, step)
    grid_lat, grid_lon = np.meshgrid(lats, lons, indexing="ij")
    place_lat = np.array([lat for _, lat, _ in PLACES])
    place_lon = np.array([lon for _, _, lon in PLACES])
    lat = np.concatenate([grid_lat.ravel(), place_lat])
    lon = np.concatenate([grid_lon.ravel(), place_lon])
    land_mask = on_land(rings, lat, lon)
    cells = grid_lat.size

    base = interpolate_track(observed, TIMES.astype(float))
    base[:, 2] *= spec["intensity"]
    growth = TIMES / 72.0
    rng = np.random.default_rng(7 if key == "full" else 11)

    tracks, winds, groups = [], [], []
    for model, (along, left, strength, trades) in enumerate(
        zip(spec["along"], spec["left"], spec["strength"], spec["trades"], strict=True)
    ):
        for _ in range(spec["runs"]):
            run_along = rng.normal(0, spec["run_along"]) * growth + rng.normal(0, 15) * growth**2
            run_left = rng.normal(0, spec["run_left"]) * growth + rng.normal(0, 12) * growth**2
            track = displace(base, TIMES.astype(float), along * growth**1.3 + run_along, left * growth**1.3 + run_left)
            track[:, 2] *= strength * (1 + rng.normal(0, spec["run_strength"]) * np.sqrt(growth))
            rmax = spec["rmax"] * (1 + rng.normal(0, 0.12))
            eddies = (0.5, rng.uniform(0, 360), rng.uniform(0, 360))
            fields = [wind(lat, lon, land_mask, track, t, rmax, trades, eddies, rng) for t in range(len(TIMES))]
            tracks.append(track)
            winds.append(fields)
            groups.append(model)

    winds = np.array(winds)  # (runs, times, cells + places)
    groups = np.array(groups)
    models = len(spec["along"])
    by_model = winds.reshape(models, spec["runs"], len(TIMES), -1)
    # the law of total variance, as probly's SecondOrderVarianceDecomposition computes it
    aleatoric = by_model.var(axis=1).mean(axis=0)
    epistemic = by_model.mean(axis=1).var(axis=0) if models > 1 else np.zeros_like(aleatoric)
    total = aleatoric + epistemic
    lower, upper = np.quantile(winds, [0.05, 0.95], axis=0)
    gale = (winds >= GALE).mean(axis=0)

    # what happened: the same wind model along the observed track, on the same grid
    observed_base = interpolate_track(observed, TIMES.astype(float))
    observed_base[:, 2] *= spec["intensity"]
    truth = np.array(
        [
            wind(
                lat,
                lon,
                land_mask,
                observed_base,
                t,
                spec["rmax"] * 0.9,
                0.0,
                (2.0, 40.0, 210.0),
                np.random.default_rng(99),
            )
            for t in range(len(TIMES))
        ]
    )
    share = np.where(total > 1e-9, epistemic / np.maximum(total, 1e-9), 0.0)
    grid = slice(0, cells)
    fields = {
        "single": [encode(winds[0, t, grid], 4) for t in range(len(TIMES))],
        "mean": [encode(winds[:, t, grid].mean(axis=0), 4) for t in range(len(TIMES))],
        "gale": [encode(gale[t, grid], 255) for t in range(len(TIMES))],
        "spread": [encode(np.sqrt(total[t, grid]), 20) for t in range(len(TIMES))],
        "epistemic": [encode(share[t, grid], 255) for t in range(len(TIMES))],
        "lower": [encode(lower[t, grid], 4) for t in range(len(TIMES))],
        "upper": [encode(upper[t, grid], 4) for t in range(len(TIMES))],
        "truth": [encode(truth[t, grid], 4) for t in range(len(TIMES))],
    }
    rows, cols = grid_lat.shape
    places = {}
    for p, (name, _, _) in enumerate(PLACES):
        column = cells + p
        places[name] = {
            "runs": np.round(winds[:, :, column], 1).tolist(),
            "truth": np.round(truth[:, column], 1).tolist(),
            "aleatoric": np.round(aleatoric[:, column], 2).tolist(),
            "epistemic": np.round(epistemic[:, column], 2).tolist(),
        }
    return {
        "name": spec["name"],
        "label": spec["label"],
        "detail": spec["detail"],
        "models": models,
        "runs_per_model": spec["runs"],
        "grid": {"west": LON[0], "north": LAT[1], "step": step, "rows": rows, "cols": cols},
        "scales": {"wind": 4, "gale": 255, "spread": 20, "epistemic": 255},
        "fields": fields,
        "run_fields": windows(
            winds[:, :, grid].reshape(len(winds), len(TIMES), rows, cols), tracks, step, spec["window"]
        ),
        "tracks": [
            {
                "model": int(g),
                "points": np.round(track, 2).tolist(),
                "landfall": landfall(rings, track),
            }
            for g, track in zip(groups, tracks, strict=True)
        ],
        "places": places,
    }


def main() -> None:
    """Write the mock data file."""
    observed = observed_track()
    rings = land()
    data = {
        "mock": True,
        "storm": "Milton",
        "init": INIT.strftime("%Y-%m-%dT%H:%MZ"),
        "times": TIMES.tolist(),
        "extent": {"west": LON[0], "east": LON[1], "south": LAT[0], "north": LAT[1]},
        "gale": round(GALE, 2),
        "land": [np.round(ring, 2).tolist() for ring in rings],
        "places": [{"name": name, "lat": lat, "lon": lon} for name, lat, lon in PLACES],
        "coast": [{"name": name, "lat": lat, "lon": lon} for name, lat, lon in COAST],
        "observed": {
            "points": [[round(h, 2), lat, lon, round(w, 1)] for h, lat, lon, w in observed],
            "landfall": {"hours": 72.5, "lat": 27.4, "lon": -82.6, "place": "Siesta Key"},
            "source": "IBTrACS v04r01 (NOAA NCEI)",
        },
        "models": {},
    }
    for key in MODELS:
        data["models"][key] = build_model(key, rings, observed)
    text = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    OUT.write_text(
        "// The data of the Milton demo, written by compute/mock.py. MOCK DATA: every forecast in it is\n"
        "// invented; only the observed track (IBTrACS) and the coastlines (Natural Earth) are real.\n"
        f"window.PROBLY_DEMO_MILTON = {text};\n",
        encoding="utf-8",
    )
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KiB)")  # noqa: T201


if __name__ == "__main__":
    main()
