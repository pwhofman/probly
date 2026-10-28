r"""Run WeatherNext Cyclones on Hurricane Milton, and write the Milton demo's data.

Downloads WeatherNext's public initial conditions of 7 October 2024, 00 UTC (0.25 degrees, 9.2 GB)
and the four public checkpoints of WeatherNext Cyclones (0.7 GB each; four models trained
independently on data through 2023, before Milton), rolls out 8 runs of 5 days of each model through
probly's WeatherNext integration, and writes the demo's data,
``docs/source/demos/milton/assets/data.js``. The runs land in ``compute/results/full`` and the
downloads in ``compute/cache/weathernext``; git ignores both. If the script stops, run it again: it
skips the runs it already has.

What to install next to probly
------------------------------
One environment, Python 3.12 or newer, with probly from this branch, JAX for the GPU, and
WeatherNext with its dependencies. WeatherNext is not on PyPI, and its ``colabtools`` dependency is
only for Colab, so it goes in without its dependencies, which are listed here instead, in the
versions the demo's runs were made with (JAX and Haiku break each other's newer releases now and
then). From the repository's root::

    uv venv --python 3.12 .venv-weathernext && source .venv-weathernext/bin/activate
    uv pip install -e .  # probly
    uv pip install "jax[cuda12]==0.12.0.1"  # JAX for NVIDIA GPUs, with its own CUDA libraries (12.9)
    uv pip install chex==0.1.92 dask==2026.8.0 dinosaur-dycore==1.2.1 dm-haiku==0.0.17 dm-tree==0.1.10 \
        fiddle==0.3.1 h5netcdf==1.8.2 h5py==3.16.0 jraph==0.0.6.dev0 matplotlib==3.11.2 \
        netcdf4==1.7.5.1.post1 numpy==2.5.4 pandas==3.1.0 pyproj==3.8.0 rtree==1.4.1 scipy==1.18.1 \
        trimesh==5.2.0 typing_extensions==4.16.0 xarray==2026.2.0 xarray_tensorstore==0.3.0 \
        "gdm-xarray-jax @ git+https://github.com/google-deepmind/xarray_jax.git@v0.1.1"
    git clone https://github.com/google-deepmind/weathernext ../weathernext
    git -C ../weathernext checkout f2f2c51
    uv pip install --no-deps -e ../weathernext

Without uv, ``python -m venv`` and ``pip install`` do the same. ``python -c "import jax;
print(jax.devices())"`` should list the GPUs.

Hardware
--------
WeatherNext says that its full models need an H100 (80 GB) on a GPU; GPUs with more memory, such as
a B100 or B200, should do too, but we have not tried. On a GPU, the attention runs as
``triblockdiag_mha``, which needs more memory than the TPU's, so this script lets JAX take 95% of a
GPU's memory instead of its default 75% (set ``XLA_PYTHON_CLIENT_MEM_FRACTION`` to change it). With
several GPUs, as many runs go at once, one per GPU. The host needs about 32 GB of memory: the 9.2
GB of initial conditions are loaded whole, the template of the forecast is about as large again,
and compiling the model takes more.

Run
---
From the repository's root, first a check (one step of one run: the downloads, then the model's
compilation and its first step; it shows whether the model fits and how long a step takes), then the
runs::

    python docs/source/demos/milton/compute/run_full.py --check
    python docs/source/demos/milton/compute/run_full.py

The runs are 4 models x 8 runs x 20 steps of 6 hours, 640 steps on one GPU; the script prints every
step with the time, and saves the runs as they are done. When it is done, commit
``docs/source/demos/milton/assets/data.js``. To look at the demo without building the docs,
``python docs/source/demos/milton/compute/preview.py`` writes ``compute/cache/preview.html``.
"""

from __future__ import annotations

import os

# before JAX starts: its default of 75% of a GPU's memory is too little for the full models
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")

from forecast import run

if __name__ == "__main__":
    run(
        "full",
        config="WeatherNextCyclones",
        data="dataset/source-hres_forecast_init-2024-10-07 00:00:00_res-0.25_levels-13_steps-20.nc",
        # four models, trained independently on data through 2023
        checkpoints={f"model{i}": f"params/WeatherNextCyclones_<2024_model{i}.npz" for i in range(1, 5)},
        description=__doc__,
    )
