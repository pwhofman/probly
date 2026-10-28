r"""Forecast Hurricane Milton with WeatherNext: the runs of the Milton demo.

Rolls out runs (ensemble members) of WeatherNext checkpoints from Milton's public initial
conditions (2024-10-07, 00 UTC) and saves every run's fields over the demo's region:
``<out>/<checkpoint>/member_<seed>.nc``, with the initial fields in ``<out>/inputs_t0.nc``.
Runs that exist are skipped, so the script can be restarted; the runs of all checkpoints alternate,
so an interrupted ensemble stays balanced. ``pack.py`` turns the runs into the demo's data.

Every run follows PR #594's example: the ensemble wrapper is removed from the config, the runs are
rolled out with WeatherNext's own ``chunked_prediction_generator_multiple_runs``, and run ``seed``
draws its noise from ``jax.random.fold_in(jax.random.PRNGKey(0), seed)``. A run is the same for
the same seed and the same number of steps (WeatherNext's noise also depends on how many steps are
rolled out). On CPUs and GPUs, the attention is ``triblockdiag_mha`` (the config's default,
``splash_mha``, needs a TPU).

This script needs WeatherNext and its dependencies, not probly's environment (see the README).
The Mini checkpoints run on a CPU (about 12 minutes per run of 20 steps on four cores, 11 GB of
memory) and on any GPU; the full ones need about 80 GB of GPU memory (an H100 or larger)::

    python forecast.py --data hres_1deg_steps20.nc \
        --checkpoint mini_2024=wn_mini_2024.npz --checkpoint mini_2023=wn_mini_2023.npz

    python forecast.py --config WeatherNextCyclones --data hres_0p25deg_steps20.nc \
        --checkpoint model1=WeatherNextCyclones_2024_model1.npz ...  # four checkpoints
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
import time

import haiku as hk
import jax
import numpy as np
from weathernext.utils import checkpoint, data_utils, fiddle_config_io, rollout
from weathernext.weathernext2 import fgn
import xarray
import xarray_jax

VARIABLES = {  # the fields the demo needs, with their units
    "10m_u_component_of_wind": "m s-1",
    "10m_v_component_of_wind": "m s-1",
    "mean_sea_level_pressure": "Pa",
    "total_precipitation_6hr": "m (6 h accumulation)",
}
LAT = slice(5, 45)  # the demo's map, with a margin for the storm's runs
LON = slice(250, 310)


def log(message: str) -> None:
    """Print a message with the time."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)  # noqa: T201


def crop(fields: xarray.Dataset, init: np.datetime64) -> xarray.Dataset:
    """The demo's fields over its region, with the valid time and units."""
    out = fields[[name for name in VARIABLES if name in fields.data_vars]].sel(lat=LAT, lon=LON)
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


def main() -> None:
    """Roll out and save the runs."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default="WeatherNextCyclones_Mini", help="a config of weathernext2/configs")
    parser.add_argument("--data", type=Path, required=True, help="WeatherNext's public initial conditions")
    parser.add_argument("--checkpoint", action="append", required=True, help="name=path, once per checkpoint")
    parser.add_argument("--runs", type=int, default=8, help="runs per checkpoint (seeds 0, 1, ...)")
    parser.add_argument("--steps", type=int, default=20, help="steps of 6 hours")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()

    batch = xarray.load_dataset(args.data).compute()
    init = np.datetime64(batch.datetime.values[0, 1], "ns")  # the second input time is the start
    config = fiddle_config_io.get_fiddle_config_by_name(f"weathernext2/configs/{args.config}")
    inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
        batch, target_lead_times=slice("6h", f"{args.steps * 6}h"), **dataclasses.asdict(config.task)
    )
    args.out.mkdir(parents=True, exist_ok=True)
    initial = batch.isel(time=[1]).assign_coords(time=np.array([0], dtype="timedelta64[ns]"))
    save(crop(initial, init), args.out / "inputs_t0.nc", source=args.data.name, init_time=init)

    transformer = config.predictor_kwargs["noisy_function_kwargs"]["mesh_model_ctor"].keywords["transformer_kwargs"]
    if jax.default_backend() != "tpu":
        transformer["attention_type"] = "triblockdiag_mha"
    log(f"{args.config} on {jax.default_backend()}: {jax.devices()}")
    inference = fgn.PredictorConfig(
        task=config.task,
        predictor_constructor=config.predictor_constructor,
        predictor_kwargs=config.predictor_kwargs,
        predictor_wrappers=config.predictor_wrappers[:-1],  # without the ensemble wrapper
    )

    @hk.transform
    def forward(inputs: xarray.Dataset, targets_template: xarray.Dataset, forcings: xarray.Dataset) -> object:
        return fgn.construct_predictor(inference)(inputs, targets_template=targets_template, forcings=forcings)

    apply = jax.jit(forward.apply)

    def runs_of(params: object) -> object:
        """A checkpoint's forecast function, mapped over runs (the parameters closed over, as in PR #594)."""
        return xarray_jax.pmap(lambda rng, i, t, f: apply(params, rng, i, t, f), dim="sample")

    predictors = {}
    for entry in args.checkpoint:
        name, path = entry.split("=", 1)
        with Path(path).open("rb") as file:
            predictors[name] = (path, runs_of(checkpoint.load(file, fgn.CheckPoint).params))

    for seed in range(args.runs):
        for name, (path, predictor) in predictors.items():
            (args.out / name).mkdir(exist_ok=True)
            out = args.out / name / f"member_{seed}.nc"
            if out.exists():
                log(f"{out} exists")
                continue
            start = time.perf_counter()
            chunks = []
            for chunk in rollout.chunked_prediction_generator_multiple_runs(
                predictor_fn=predictor,
                rngs=np.stack([jax.random.fold_in(jax.random.PRNGKey(0), seed)]),
                inputs=inputs,
                targets_template=targets * np.nan,
                forcings=forcings,
                num_steps_per_chunk=1,
                num_samples=1,
                pmap_devices=jax.local_devices(),
            ):
                # block, and move the chunk to the host (see PR #594)
                jax.block_until_ready([xarray_jax.unwrap_data(chunk[var]) for var in chunk.data_vars])
                chunks.append(chunk.as_numpy())
            run = crop(xarray.combine_by_coords(chunks), init)
            save(
                run,
                out,
                model=args.config,
                checkpoint=Path(path).name,
                seed=seed,
                rng="jax.random.fold_in(jax.random.PRNGKey(0), seed)",
                attention=transformer["attention_type"],
                init_time=init,
                initial_conditions=args.data.name,
            )
            log(f"{out}: {len(chunks)} steps in {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
