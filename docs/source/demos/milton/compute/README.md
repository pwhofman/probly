# How the Milton demo's data is made

The Milton demo (`../assets/`: `demo.js`, `demo.css` and `data.js`) shows WeatherNext's runs of
Hurricane Milton, initialized on 7 October 2024, 00 UTC, over four days, and what probly makes of
their spread: the split into aleatoric and epistemic uncertainty, with probly's
`SecondOrderVarianceDecomposition`. The data is computed once, by hand, and committed; the page only
looks values up.

**Which runs are real.** The Mini model's runs are WeatherNext's: 8 runs of each of the two public
Mini checkpoints, computed on a CPU, which stand in for WeatherNext's four models. The full model's
runs are still mock data, invented around Milton's real track: the full checkpoints need a GPU with
about 80 GB of memory (an H100 or larger) or a TPU. Milton's observed track (IBTrACS) and the
coastlines (Natural Earth) are real.

| File | What it does | Environment |
|---|---|---|
| `forecast.py` | Rolls out WeatherNext's runs of Milton: `results/<checkpoint>/member_<seed>.nc`. | WeatherNext's |
| `pack.py` | Writes `../assets/data.js` from the runs, with mock runs for the models that have none. | probly's, with xarray |
| `mock.py` | The mock runs; run alone, it writes `../assets/data.js` with mock runs only. | probly's |
| `common.py` | What the scripts share: the map, the places, the observed track, the data's format. | probly's |
| `preview.py` | Inlines the assets into one page, `cache/preview.html`, to look at the demo without the docs. | any |

## Run

**The runs.** WeatherNext is not on PyPI and needs Python 3.12 or newer, in an environment of its own:

```bash
git clone https://github.com/google-deepmind/weathernext
uv venv --python 3.12 wn-venv && source wn-venv/bin/activate
uv pip install jax chex dask dinosaur-dycore dm-haiku dm-tree fiddle h5netcdf h5py jraph matplotlib \
    numpy pandas pyproj rtree scipy trimesh typing_extensions "xarray<=2026.2.0" xarray_tensorstore \
    "gdm-xarray-jax @ git+https://github.com/google-deepmind/xarray_jax.git@v0.1.1" netcdf4
uv pip install --no-deps -e weathernext  # use "jax[cuda12]" above on a GPU

BASE=https://storage.googleapis.com/dm_graphcast/weathernext2
curl -o wn_mini_2024.npz "$BASE/params/WeatherNextCyclones_Mini_%3C2024.npz"
curl -o wn_mini_2023.npz "$BASE/params/WeatherNextCyclones_Mini_%3C2023.npz"
curl -o hres_1deg_steps20.nc \
    "$BASE/dataset/source-hres_forecast_init-2024-10-07%2000%3A00%3A00_res-1.0_levels-13_steps-20.nc"

python forecast.py --data hres_1deg_steps20.nc \
    --checkpoint mini_2024=wn_mini_2024.npz --checkpoint mini_2023=wn_mini_2023.npz
```

On four CPU cores, a Mini run of 20 steps (5 days) takes about 12 minutes and 11 GB of memory, so
the 16 runs take about 3 hours; on a GPU, they take minutes. The full model runs the same way with
`--config WeatherNextCyclones`, the 0.25° data and its four checkpoints
(`WeatherNextCyclones_%3C2024_model{1,2,3,4}.npz`, trained on data through 2023, before Milton).

**The demo's data**, from the repository's root:

```bash
uv run --with xarray --with h5netcdf --with h5py python docs/source/demos/milton/compute/pack.py \
    --results docs/source/demos/milton/compute/results
uv run python docs/source/demos/milton/compute/preview.py  # then open cache/preview.html
```

`results/` and `cache/` are not part of the repository.

## The data

`data.js` sets `window.PROBLY_DEMO_MILTON`: the lead times (every 6 hours, 0 to 96 hours), the
coastlines, the places, Milton's observed track, and per model (`full`, `mini`), with `mock` saying
whether its runs are invented:

- `fields`: per lead time, the grid's values as bytes in base64 (rows north to south): one run's
  wind (`single`), the runs' mean, the chance of tropical-storm-force wind (`gale`), the spread
  (`spread`, the standard deviation of the runs), the epistemic share of its variance
  (`epistemic`), the runs' 5% and 95% quantiles (`lower`, `upper`) and, where there is one, what
  happened (`truth`);
- `run_fields`: every run's wind near the storm, per lead time a square of cells that follows the
  runs' mean position (`origins`: its first row and column), for the outline of every run's
  tropical-storm-force wind;
- `tracks`: every run's track (the lowest sea-level pressure near its last position) and where it
  comes ashore in Florida;
- `places`: every run's wind at the demo's places, with the aleatoric and epistemic variance.

What happened to the wind is still missing for the real runs: the targets in WeatherNext's public
data are ECMWF's HRES forecast from the same start, not observations. The observed wind will come
from a reanalysis (ERA5).
