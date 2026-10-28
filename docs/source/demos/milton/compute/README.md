# How the Milton demo's data is made

The Milton demo (`../assets/`: `demo.js`, `demo.css` and `data.js`) shows WeatherNext's runs of
Hurricane Milton, initialized on 7 October 2024, 00 UTC, over four days, and what probly makes of
their spread: the split into aleatoric and epistemic uncertainty, with probly's
`SecondOrderVarianceDecomposition`. The data is computed once, by hand, and committed; the page only
looks values up.

**Which runs are real.** The Mini model's runs are WeatherNext's: 8 runs of each of the two public
Mini checkpoints, computed on a CPU, which stand in for WeatherNext's four models. The full model's
runs are placeholders, made up around Milton's real track (`mock_runs.py`), until `run_full.py` has
run on a GPU with about 80 GB of memory (an H100 or larger). Milton's observed track (IBTrACS) and
the coastlines (Natural Earth) are real.

| File | What it does | Environment |
|---|---|---|
| `run_mini.py`, `run_full.py` | Start here: download WeatherNext's public files, roll out the Mini or the full model's runs, and write `../assets/data.js`. Their docstrings say what to install and how long it takes. | WeatherNext's, with probly |
| `forecast.py` | Rolls out WeatherNext's runs of Milton through probly's `WeatherNextPredictor` and `representer`: `results/<model>/<checkpoint>/member_<seed>.nc`. | WeatherNext's, with probly |
| `pack.py` | Writes `../assets/data.js` from the runs; a model without runs keeps its real runs from `data.js`, or else gets mock runs. | WeatherNext's, or probly's with xarray |
| `mock_runs.py` | The mock runs; run alone, it writes `../assets/data.js` with mock runs only. | probly's |
| `common.py` | What the scripts share: the map, the places, the observed track, the data's format. | probly's |
| `preview.py` | Inlines the assets into one page, `cache/preview.html`, to look at the demo without the docs. | any |
| `fallback.py` | Photographs the demo in the preview for its fallback image, `../../../_static/demos/milton.png`, shown without JavaScript and on the Demos page. Run it when the demo or its data changed. | probly's, with Playwright |

## Run

From the repository's root, in one environment with probly, JAX and WeatherNext (the scripts'
docstrings say how to set it up):

```bash
python docs/source/demos/milton/compute/run_mini.py --check  # one step of one run, to test the setup
python docs/source/demos/milton/compute/run_mini.py  # the Mini runs: 3 hours on four CPU cores
python docs/source/demos/milton/compute/run_full.py --check
python docs/source/demos/milton/compute/run_full.py  # the full runs: a GPU with 80 GB (an H100 or larger)
python docs/source/demos/milton/compute/preview.py  # then open cache/preview.html
```

Each script downloads what it needs into `cache/weathernext`, writes its runs to `results/mini` or
`results/full` as they are done, skips what already exists when restarted, and writes
`../assets/data.js` at the end, with the other model's real runs kept; it stops with an error if its
own runs did not make it into the data. `results/` and `cache/` are not part of the repository;
commit `../assets/data.js`, and then the fallback image. To pack the runs again without rolling them
out, and to photograph the demo:

```bash
uv run --with xarray --with h5netcdf --with h5py python docs/source/demos/milton/compute/pack.py
uv run --with playwright playwright install chromium  # once
uv run --with playwright python docs/source/demos/milton/compute/fallback.py
```

## The data

`data.js` sets `window.PROBLY_DEMO_MILTON`: the lead times (every 6 hours, 0 to 96 hours), the
coastlines, the places, Milton's observed track, and per model (`full`, `mini`), with `mock` saying
whether its runs are placeholders. Fields are bytes, value times the model's `scales` (wind up to 85
m/s, the spread up to 25.5 m/s); `pack.py` warns when a value does not fit.

- `fields`: per lead time, the grid's values as bytes in base64 (rows north to south): one run's
  wind (`single`), the runs' mean, the chance of tropical-storm-force wind (`gale`), the spread
  (`spread`, the standard deviation of the runs), the epistemic share of its variance
  (`epistemic`), the runs' 5% and 95% quantiles (`lower`, `upper`) and, where there is one, what
  happened (`truth`);
- `run_fields`: every run's wind near the storm, per lead time a window of `rows` x `cols` cells
  around where any run has tropical-storm-force wind, with a cell to spare (`origins`: its first
  row and column), for the outline around it;
- `tracks`: every run's track (the lowest sea-level pressure near its last position) and where it
  comes ashore in Florida;
- `places`: every run's wind at the demo's places, with the aleatoric and epistemic variance.

What happened to the wind is still missing for the real runs: the targets in WeatherNext's public
data are ECMWF's HRES forecast from the same start, not observations. The observed wind will come
from a reanalysis (ERA5).
