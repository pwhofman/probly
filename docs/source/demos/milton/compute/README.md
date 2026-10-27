# How the Milton demo's data is made

The Milton demo (`../assets/`: `demo.js`, `demo.css` and `data.js`) shows WeatherNext's runs of
Hurricane Milton, initialized on 7 October 2024, 00 UTC, over four days, and what probly makes of
their spread: the split into aleatoric and epistemic uncertainty, and conformal ranges. The data is
computed once, by hand, and committed; the page only looks values up.

**For now, the data is mock data.** `mock.py` writes `../assets/data.js` with invented runs around
Milton's real track, in the format the real pipeline will write, so that the demo can be designed and
reviewed before the forecasts exist. Only the observed track (IBTrACS) and the coastlines (Natural
Earth) in it are real.

| File | What it does |
|---|---|
| `mock.py` | Writes `../assets/data.js` with mock data (downloads IBTrACS and Natural Earth into `cache/` once). |
| `preview.py` | Inlines the assets into one page, `cache/preview.html`, to look at the demo without building the docs. |

## Run

From the repository's root:

```bash
uv run python docs/source/demos/milton/compute/mock.py
uv run python docs/source/demos/milton/compute/preview.py  # then open cache/preview.html
```

## The data, and where the real one will come from

`data.js` sets `window.PROBLY_DEMO_MILTON`: the lead times (every 6 hours, 0 to 96 hours), the
coastlines, the places, Milton's observed track, and per model (`full`, `mini`):

- `fields`: per lead time, the grid's values as bytes in base64 (rows north to south): one run's
  wind (`single`), the runs' mean, the chance of tropical-storm-force wind (`gale`), the spread
  (`spread`, the standard deviation of the runs), the epistemic share of its variance
  (`epistemic`), the runs' 5% and 95% quantiles (`lower`, `upper`), and what happened (`truth`);
- `run_fields`: every run's wind near the storm, per lead time a square of cells that follows the
  runs' mean position (`origins`: its first row and column), for the contour lines of every run;
- `tracks`: every run's track and where it comes ashore in Florida;
- `places`: every run's wind at the demo's places, with the aleatoric and epistemic variance;
- `conformal`: the CQR correction of the runs' 90% range per lead time, calibrated on half of the
  map, and the coverage on the other half before and after.

The real data will come from WeatherNext's public checkpoints and Milton's public initial
conditions (`gs://dm_graphcast/weathernext2/`), through probly's WeatherNext integration (PR #594):

- `full`: `WeatherNextCyclones_<2024_model{1,2,3,4}` (0.25°, trained on data through 2023, so
  before Milton), 8 runs per model, on a GPU;
- `mini`: `WeatherNextCyclones_Mini_<2024` (1°), 8 runs, which also runs on a CPU;
- `truth`: the HRES analyses of the same file, the targets of PR #594's example.
