"""What the Milton demo's data scripts share: the map, the places, Milton's observed track, the format.

``mock_runs.py`` (made-up runs) and ``pack.py`` (WeatherNext's runs) both summarize runs with
:func:`summarize` and write ``../assets/data.js`` with :func:`write`, so that the demo reads the same
format from both. The runs' spread is split with probly's ``SecondOrderVarianceDecomposition``: the
runs of every model are a Gaussian (their mean and variance), and the models are a sample of them.
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

from probly.quantification.decomposition.variance import SecondOrderVarianceDecomposition
from probly.representation.distribution.numpy_gaussian import (
    NumpyGaussianDistribution,
    NumpyGaussianDistributionSample,
)

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
OUT = HERE.parent / "assets" / "data.js"

IBTRACS_URL = (
    "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/"
    "v04r01/access/csv/ibtracs.NA.list.v04r01.csv"
)
LAND_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/v5.1.2/geojson/ne_50m_land.geojson"
MILTON = "2024279N21265"  # Milton's IBTrACS storm id

INIT = datetime(2024, 10, 7, tzinfo=UTC)  # the initialization of the public WeatherNext data
TIMES = np.arange(0, 97, 6)  # lead times in hours, every 6 h for four days
LON = (-94.5, -72.0)  # the map: the Gulf of Mexico, Florida and the western Atlantic
LAT = (20.3, 33.0)  # just north of 20 degrees N, so that its line is not drawn at the map's edge
GRID_LON = (-95.0, -72.0)  # the grid of the fields: the map, out to whole degrees
GRID_LAT = (20.0, 33.0)
KNOT = 0.514444  # m/s
GALE = 34 * KNOT  # tropical-storm-force wind, NHC's lowest wind speed probability threshold
# the fields are bytes: value * scale, so wind up to 85 m/s, spread up to 25.5 m/s, shares to 1
SCALES = {"wind": 3, "gale": 255, "spread": 10, "epistemic": 255}
FLORIDA = (28.54, -81.38)  # Orlando: the land around it is Florida's mainland

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
    ("Tampa Bay", 27.77, -82.68),  # at its mouth, by St. Petersburg
    ("Clearwater", 27.97, -82.80),
    ("Tarpon Springs", 28.15, -82.76),
    ("Hernando Beach", 28.48, -82.66),
    ("Crystal River", 28.90, -82.63),
    ("Cedar Key", 29.14, -83.03),
]


def download(url: str, name: str) -> Path:
    """Download a file into the cache once (under another name until it is complete)."""
    path = CACHE / name
    if not path.exists():
        CACHE.mkdir(exist_ok=True)
        print(f"downloading {url}")  # noqa: T201
        partial = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(url, partial)  # noqa: S310
        partial.replace(path)
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


def grid(step: float) -> dict:
    """The demo's grid at a resolution: its first cell (north-west), its step, its rows and columns."""
    rows = round((GRID_LAT[1] - GRID_LAT[0]) / step) + 1
    cols = round((GRID_LON[1] - GRID_LON[0]) / step) + 1
    return {"west": GRID_LON[0], "north": GRID_LAT[1], "step": step, "rows": rows, "cols": cols}


def points(spec: dict) -> tuple[np.ndarray, np.ndarray]:
    """The latitudes and longitudes the demo needs on a grid: its cells (north to south), then the places."""
    lats = spec["north"] - np.arange(spec["rows"]) * spec["step"]
    lons = spec["west"] + np.arange(spec["cols"]) * spec["step"]
    grid_lat, grid_lon = np.meshgrid(lats, lons, indexing="ij")
    lat = np.concatenate([grid_lat.ravel(), [lat for _, lat, _ in PLACES]])
    lon = np.concatenate([grid_lon.ravel(), [lon for _, _, lon in PLACES]])
    return lat, lon


def landfall(rings: list[np.ndarray], track: np.ndarray) -> dict | None:
    """Where and when a track first comes ashore in Florida (half-hourly).

    Florida is the mainland (the land around Orlando, not the Keys or the Bahamas) east of Alabama,
    south of Georgia (31 degrees N) and north of the Yucatan, which are part of the same land.
    """
    mainland = next(Polygon(ring) for ring in rings if Polygon(ring).contains_point((FLORIDA[1], FLORIDA[0])))
    hours = np.arange(TIMES[0], TIMES[-1] + 0.25, 0.5)
    lat = np.interp(hours, TIMES, track[:, 0])
    lon = np.interp(hours, TIMES, track[:, 1])
    ashore = mainland.contains_points(np.column_stack([lon, lat])) & (lon > -87.6) & (lat > 24.6) & (lat < 31)
    if not ashore.any():
        return None
    first = int(np.argmax(ashore))
    return {"hours": float(hours[first]), "lat": round(float(lat[first]), 2), "lon": round(float(lon[first]), 2)}


def encode(values: np.ndarray, kind: str) -> str:
    """Values on the grid as bytes (value * the kind's scale, rounded and clipped), in base64."""
    scaled = np.round(values * SCALES[kind])
    if scaled.max(initial=0) > 255:
        print(f"warning: {kind} up to {values.max():.1f} is cut at {255 / SCALES[kind]}")  # noqa: T201
    return base64.b64encode(np.clip(scaled, 0, 255).astype(np.uint8).tobytes()).decode()


def windows(winds: np.ndarray, tracks: list[np.ndarray], spec: dict, size: int) -> dict:
    """Every run's wind near the storm, for the outline around where any run has tropical-storm-force wind.

    Per lead time a window of cells around where any run has tropical-storm-force wind, with a cell
    to spare (or around the runs' mean position, where none has), as large as it needs to be at any
    time and at least ``size`` x ``size`` cells; ``winds`` has the shape (runs, times, rows, cols).
    """
    rows, cols = winds.shape[2:]
    gale = (winds >= GALE).any(axis=0)
    boxes = []  # per lead time: first and last row, first and last column
    for t in range(len(TIMES)):
        r, c = np.nonzero(gale[t])
        if len(r):
            boxes.append((r.min() - 1, r.max() + 1, c.min() - 1, c.max() + 1))
        else:
            lat, lon = np.mean([track[t] for track in tracks], axis=0)[:2]
            row = round((spec["north"] - lat) / spec["step"])
            col = round((lon - spec["west"]) / spec["step"])
            boxes.append((row, row, col, col))
    height = int(min(rows, max(size, *(last - first + 1 for first, last, _, _ in boxes))))
    width = int(min(cols, max(size, *(last - first + 1 for _, _, first, last in boxes))))
    origins, fields = [], []
    for t, (top, bottom, left, right) in enumerate(boxes):
        row = int(np.clip((top + bottom + 1) // 2 - height // 2, 0, rows - height))
        col = int(np.clip((left + right + 1) // 2 - width // 2, 0, cols - width))
        origins.append([row, col])
        fields.append(encode(winds[:, t, row : row + height, col : col + width], "wind"))
    return {"rows": height, "cols": width, "origins": origins, "fields": fields}


def decompose(winds: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The total, aleatoric and epistemic variance of runs of shape (models, runs, ...), with probly.

    The runs of every model are a Gaussian with their mean and variance; the models are a sample of
    these, a second-order distribution, whose variance probly splits by the law of total variance.
    """
    # at the start, all runs are the same analysis: their variance is zero, which a Gaussian does not
    # take, so it gets a floor far below what the demo shows
    gaussians = NumpyGaussianDistribution(mean=winds.mean(axis=1), var=np.maximum(winds.var(axis=1), 1e-9))
    split = SecondOrderVarianceDecomposition(NumpyGaussianDistributionSample(array=gaussians, sample_axis=0))
    return np.asarray(split.total), np.asarray(split.aleatoric), np.asarray(split.epistemic)


def summarize(
    winds: np.ndarray,
    truth: np.ndarray | None,
    tracks: np.ndarray,
    rings: list[np.ndarray],
    spec: dict,
    window: int,
) -> dict:
    """Runs summarized as the demo needs them.

    Args:
        winds: The runs' wind speed (m/s) of shape (models, runs, times, cells + places), at the
            cells of ``spec`` (north to south, west to east) and then at the places.
        truth: What happened, of shape (times, cells + places), or ``None`` if there is no truth.
        tracks: The runs' tracks of shape (models, runs, times, 3): latitude, longitude (west of
            Greenwich negative) and the maximum wind near the center.
        rings: The land, for the landfalls.
        spec: The grid (:func:`grid`).
        window: The size of every run's square of cells near the storm.

    Returns:
        The model's entry in the demo's data, but for its name and description.
    """
    models, runs_per_model, times = winds.shape[:3]
    cells = spec["rows"] * spec["cols"]
    total, aleatoric, epistemic = decompose(winds)
    runs = winds.reshape(models * runs_per_model, times, -1)
    lower, upper = np.quantile(runs, [0.05, 0.95], axis=0)
    gale = (runs >= GALE).mean(axis=0)
    share = np.where(total > 1e-9, epistemic / np.maximum(total, 1e-9), 0.0)
    flat_tracks = tracks.reshape(models * runs_per_model, times, 3)
    on_grid = slice(0, cells)
    frames = range(times)
    fields = {
        "single": [encode(runs[0, t, on_grid], "wind") for t in frames],
        "mean": [encode(runs[:, t, on_grid].mean(axis=0), "wind") for t in frames],
        "gale": [encode(gale[t, on_grid], "gale") for t in frames],
        "spread": [encode(np.sqrt(total[t, on_grid]), "spread") for t in frames],
        "epistemic": [encode(share[t, on_grid], "epistemic") for t in frames],
        "lower": [encode(lower[t, on_grid], "wind") for t in frames],
        "upper": [encode(upper[t, on_grid], "wind") for t in frames],
    }
    if truth is not None:
        fields["truth"] = [encode(truth[t, on_grid], "wind") for t in frames]
    places = {}
    for p, (name, _, _) in enumerate(PLACES):
        column = cells + p
        places[name] = {
            "runs": np.round(runs[:, :, column], 1).tolist(),
            "aleatoric": np.round(aleatoric[:, column], 2).tolist(),
            "epistemic": np.round(epistemic[:, column], 2).tolist(),
        }
        if truth is not None:
            places[name]["truth"] = np.round(truth[:, column], 1).tolist()
    return {
        "models": models,
        "runs_per_model": runs_per_model,
        "grid": spec,
        "scales": SCALES,
        "fields": fields,
        "run_fields": windows(
            runs[:, :, on_grid].reshape(len(runs), times, spec["rows"], spec["cols"]),
            list(flat_tracks),
            spec,
            window,
        ),
        "tracks": [
            {"model": run // runs_per_model, "points": np.round(track, 2).tolist(), "landfall": landfall(rings, track)}
            for run, track in enumerate(flat_tracks)
        ],
        "places": places,
    }


def write(models: dict, observed: list, rings: list[np.ndarray], note: str) -> None:
    """Write the demo's data, ``../assets/data.js``, with a note on where it comes from."""
    data = {
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
        "models": models,
    }
    text = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    comment = "".join(f"// {line}\n" for line in note.splitlines())
    OUT.write_text(f"{comment}window.PROBLY_DEMO_MILTON = {text};\n", encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KiB)")  # noqa: T201
