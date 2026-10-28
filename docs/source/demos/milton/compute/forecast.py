r"""Forecast Hurricane Milton with WeatherNext: the runs of the Milton demo.

Rolls out runs (ensemble members) of WeatherNext checkpoints from Milton's public initial
conditions (2024-10-07, 00 UTC) and saves every run's fields over the demo's region:
``<out>/<checkpoint>/member_<seed>.nc``, with the initial fields in ``<out>/inputs_t0.nc``.
A checkpoint whose runs all exist is skipped, so the script can be restarted. ``pack.py`` turns the
runs into the demo's data. ``run_mini.py`` and ``run_full.py`` do both, with the public files: start
there.

The runs go through probly's WeatherNext integration: every checkpoint is a ``WeatherNextPredictor``,
and ``representer(predictor, num_samples=runs)`` collects its runs, one sample per variable. The
ensemble wrapper is removed from the config, the runs are rolled out with WeatherNext's own
``chunked_prediction_generator_multiple_runs``, and run ``seed`` draws its noise from
``jax.random.fold_in(jax.random.PRNGKey(0), seed)``. A run is the same for the same seed and the same
number of steps (WeatherNext's noise also depends on how many steps are rolled out). On CPUs and GPUs,
the attention is ``triblockdiag_mha`` (the config's default, ``splash_mha``, needs a TPU).

This script needs WeatherNext and its dependencies, with probly installed next to them (see
``run_mini.py``). The Mini checkpoints run on a CPU (about 12 minutes per run of 20 steps on four
cores, 11 GB of memory) and on any GPU; the full ones need a GPU with about 80 GB of memory (an H100
or larger) or a TPU::

    python forecast.py --data hres_1deg_steps20.nc --out results/mini \
        --checkpoint mini_2024=wn_mini_2024.npz --checkpoint mini_2023=wn_mini_2023.npz

    python forecast.py --config WeatherNextCyclones --data hres_0p25deg_steps20.nc --out results/full \
        --checkpoint model1=WeatherNextCyclones_2024_model1.npz ...  # four checkpoints
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
import shutil
import time
from typing import TYPE_CHECKING
import urllib.parse
import urllib.request

import haiku as hk
import jax
import numpy as np
from pack import pack
from weathernext.utils import checkpoint, data_utils, fiddle_config_io, rollout
from weathernext.weathernext2 import fgn
import xarray
import xarray_jax

from probly.integrations.weathernext import WeatherNextPredictor
from probly.representer import representer

if TYPE_CHECKING:
    from collections.abc import Callable

    from probly.representation.sample.numpy import NumpySample

HERE = Path(__file__).resolve().parent
BUCKET = "https://storage.googleapis.com/dm_graphcast/weathernext2"  # WeatherNext's public files
DOWNLOADS = HERE / "cache" / "weathernext"
VARIABLES = {  # the fields the demo needs, with their units
    "10m_u_component_of_wind": "m s-1",
    "10m_v_component_of_wind": "m s-1",
    "mean_sea_level_pressure": "Pa",
    "total_precipitation_6hr": "m (6 h accumulation)",
}
LAT = (5, 45)  # the demo's map, with a margin for the storm's runs
LON = (250, 310)


def log(message: str) -> None:
    """Print a message with the time."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)  # noqa: T201


def public_file(name: str, folder: Path = DOWNLOADS) -> Path:
    """A file of WeatherNext's public bucket (e.g. ``params/...``), downloaded into ``folder`` once."""
    path = folder / name.replace("<", "").replace(" ", "_").replace(":", "")
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BUCKET}/{urllib.parse.quote(name)}"
    log(f"downloading {url}")
    shown = [0]

    def progress(blocks: int, size: int, total: int) -> None:
        done = blocks * size
        if total > 0 and done * 10 // total > shown[0]:
            shown[0] = done * 10 // total
            log(f"  {min(done, total) / 1e9:.2f} of {total / 1e9:.2f} GB")

    partial = path.with_suffix(path.suffix + ".part")
    urllib.request.urlretrieve(url, partial, progress)  # noqa: S310
    partial.replace(path)
    return path


def region(fields: xarray.Dataset) -> xarray.Dataset:
    """The demo's fields over its region, latitudes ascending."""
    out = fields[[name for name in VARIABLES if name in fields.data_vars]]
    ascending = out.lat.values[0] < out.lat.values[-1]
    out = out.sel(lat=slice(*LAT) if ascending else slice(*LAT[::-1]), lon=slice(*LON))
    return out if ascending else out.sortby("lat")


def crop(fields: xarray.Dataset, init: np.datetime64) -> xarray.Dataset:
    """The demo's fields over its region, with the valid time and units."""
    out = region(fields)
    for dim in ("batch", "sample"):
        if dim in out.dims:
            out = out.isel({dim: 0})
    out = out.drop_vars([name for name in ("batch", "sample", "datetime") if name in out.coords])
    out = out.assign_coords(datetime=("time", init + out.time.values.astype("timedelta64[ns]")))
    for name, units in VARIABLES.items():
        if name in out:
            out[name].attrs["units"] = units
    return out.load()


def save(fields: xarray.Dataset, path: Path, **attrs: object) -> None:
    """Save fields as netCDF, atomically."""
    fields.attrs.update({key: str(value) for key, value in attrs.items()})
    temporary = path.with_suffix(".tmp")
    fields.to_netcdf(temporary)
    temporary.replace(path)


def run_of(samples: dict[str, NumpySample], template: xarray.Dataset, seed: int) -> xarray.Dataset:
    """One run's fields from probly's samples (one per variable, runs first), with the template's coordinates."""
    fields = {}
    for variable in template.data_vars:
        values = samples[variable].array[seed]
        if values.shape != template[variable].shape:
            msg = f"{variable}: the run has the shape {values.shape}, the template {template[variable].shape}"
            raise ValueError(msg)
        fields[variable] = (template[variable].dims, values)
    return xarray.Dataset(fields, coords=template.coords)


def model(name: str) -> tuple[object, str, Callable[..., object]]:
    """A WeatherNext config by name, its attention, and its jitted forward pass without the ensemble wrapper."""
    config = fiddle_config_io.get_fiddle_config_by_name(f"weathernext2/configs/{name}")
    transformer = config.predictor_kwargs["noisy_function_kwargs"]["mesh_model_ctor"].keywords["transformer_kwargs"]
    if jax.default_backend() != "tpu":
        transformer["attention_type"] = "triblockdiag_mha"
    inference = fgn.PredictorConfig(
        task=config.task,
        predictor_constructor=config.predictor_constructor,
        predictor_kwargs=config.predictor_kwargs,
        predictor_wrappers=config.predictor_wrappers[:-1],  # without the ensemble wrapper
    )

    @hk.transform
    def forward(inputs: xarray.Dataset, targets_template: xarray.Dataset, forcings: xarray.Dataset) -> object:
        return fgn.construct_predictor(inference)(inputs, targets_template=targets_template, forcings=forcings)

    return config, transformer["attention_type"], jax.jit(forward.apply)


def ensemble_of(apply: Callable[..., object], params: object) -> Callable[..., xarray.Dataset]:
    """A checkpoint's ensemble function for probly's ``WeatherNextPredictor``: the runs of the given seeds.

    Every 6-hour step is cut to the demo's fields and region as it comes, so that all runs of a checkpoint
    fit into a small machine's memory.
    """
    runs = xarray_jax.pmap(lambda rng, i, t, f: apply(params, rng, i, t, f), dim="sample")

    def ensemble(
        seeds: list[int], inputs: xarray.Dataset, targets_template: xarray.Dataset, forcings: xarray.Dataset
    ) -> xarray.Dataset:
        chunks = []
        # the runs go in groups of as many as there are devices, each group step by step
        total = targets_template.sizes["time"] * -(-len(seeds) // jax.local_device_count())
        for chunk in rollout.chunked_prediction_generator_multiple_runs(
            predictor_fn=runs,
            rngs=np.stack([jax.random.fold_in(jax.random.PRNGKey(0), seed) for seed in seeds]),
            inputs=inputs,
            targets_template=targets_template,
            forcings=forcings,
            num_steps_per_chunk=1,
            num_samples=len(seeds),
            pmap_devices=jax.local_devices(),
        ):
            # wait for the step, and move it to the host
            jax.block_until_ready([xarray_jax.unwrap_data(chunk[var]) for var in chunk.data_vars])
            chunks.append(region(chunk.as_numpy()))
            log(f"step {len(chunks)} of {total}")
        return xarray.combine_by_coords(chunks).isel(batch=0).transpose("sample", "time", ..., "lat", "lon")

    return ensemble


def forecast(config: str, data: Path, checkpoints: dict[str, Path], runs: int, steps: int, out: Path) -> None:
    """Roll out ``runs`` runs of ``steps`` steps of 6 hours of every checkpoint, and save them in ``out``."""
    batch = xarray.load_dataset(data).compute()
    init = np.datetime64(batch.datetime.values[0, 1], "ns")  # the second input time is the start
    settings, attention, apply = model(config)
    inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
        batch, target_lead_times=slice("6h", f"{steps * 6}h"), **dataclasses.asdict(settings.task)
    )
    out.mkdir(parents=True, exist_ok=True)
    initial = batch.isel(time=[1]).assign_coords(time=np.array([0], dtype="timedelta64[ns]"))
    save(crop(initial, init), out / "inputs_t0.nc", source=data.name, init_time=init)
    log(f"{config} on {jax.default_backend()}: {jax.devices()}")

    # every run's fields come back from probly's representer as one sample per variable, without
    # coordinates: the targets' template gives them back
    template = region(targets).isel(batch=0).transpose("time", ..., "lat", "lon")
    for name, path in checkpoints.items():
        (out / name).mkdir(exist_ok=True)
        files = [out / name / f"member_{seed}.nc" for seed in range(runs)]
        if all(file.exists() for file in files):
            log(f"{name}: the runs exist")
            continue
        with path.open("rb") as file:
            predictor = WeatherNextPredictor(
                ensemble_fn=ensemble_of(apply, checkpoint.load(file, fgn.CheckPoint).params)
            )
        start = time.perf_counter()
        samples = representer(predictor, num_samples=runs).represent(inputs, targets * np.nan, forcings)
        log(f"{name}: {runs} runs in {time.perf_counter() - start:.0f} s")
        for seed, file in enumerate(files):
            save(
                crop(run_of(samples, template, seed), init),
                file,
                model=config,
                checkpoint=path.name,
                seed=seed,
                rng="jax.random.fold_in(jax.random.PRNGKey(0), seed)",
                attention=attention,
                init_time=init,
                initial_conditions=data.name,
                computed_with="probly.integrations.weathernext.WeatherNextPredictor and probly.representer",
            )


def run(model: str, config: str, data: str, checkpoints: dict[str, str], description: str) -> None:
    """Roll out a model of the demo from WeatherNext's public files, and write the demo's data.

    Downloads the initial conditions ``data`` and the checkpoints (name: file in WeatherNext's bucket),
    rolls out 8 runs of 20 steps of every checkpoint into ``results/<model>``, and writes the demo's data
    with ``pack.py``. With ``--check``, rolls out one step of one run of the first checkpoint into
    ``cache/check/<model>`` instead, to see that the model fits and how long a step takes.
    """
    parser = argparse.ArgumentParser(description=description.split("\n\n", maxsplit=1)[0])
    parser.add_argument("--check", action="store_true", help="roll out one step of one run only, to test the setup")
    parser.add_argument("--downloads", type=Path, default=DOWNLOADS, help="where to keep WeatherNext's public files")
    args = parser.parse_args()
    initial = public_file(data, args.downloads)
    files = {name: public_file(path, args.downloads) for name, path in checkpoints.items()}
    if args.check:
        first = next(iter(files))
        check = HERE / "cache" / "check" / model
        shutil.rmtree(check, ignore_errors=True)  # a check always runs
        forecast(config, initial, {first: files[first]}, runs=1, steps=1, out=check)
        log("the check went through: run the script without --check for the runs")
        return
    forecast(config, initial, files, runs=8, steps=20, out=HERE / "results" / model)
    pack(HERE / "results")
    log("done: commit docs/source/demos/milton/assets/data.js")


def main() -> None:
    """Roll out and save the runs."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default="WeatherNextCyclones_Mini", help="a config of weathernext2/configs")
    parser.add_argument("--data", type=Path, required=True, help="WeatherNext's public initial conditions")
    parser.add_argument("--checkpoint", action="append", required=True, help="name=path, once per checkpoint")
    parser.add_argument("--runs", type=int, default=8, help="runs per checkpoint (seeds 0, 1, ...)")
    parser.add_argument("--steps", type=int, default=20, help="steps of 6 hours")
    parser.add_argument("--out", type=Path, required=True, help="the folder of the runs, e.g. results/mini")
    args = parser.parse_args()
    checkpoints = {name: Path(path) for name, path in (entry.split("=", 1) for entry in args.checkpoint)}
    forecast(args.config, args.data, checkpoints, args.runs, args.steps, args.out)


if __name__ == "__main__":
    main()
