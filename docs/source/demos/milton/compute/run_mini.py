r"""Run WeatherNext Cyclones Mini on Hurricane Milton, and write the Milton demo's data.

Downloads WeatherNext's public initial conditions of 7 October 2024, 00 UTC (1 degree, 0.6 GB) and the
two public Mini checkpoints (0.2 GB each; trained on data through 2023 and through 2022, before
Milton), rolls out 8 runs of 5 days of each checkpoint through probly's WeatherNext integration, and
writes the demo's data, ``docs/source/demos/milton/assets/data.js``. The runs land in
``compute/results/mini`` and the downloads in ``compute/cache/weathernext``; git ignores both. If the
script stops, run it again: it skips the runs it already has.

What to install next to probly
------------------------------
One environment, Python 3.12 or newer, with probly from this branch, JAX, and WeatherNext with its
dependencies. WeatherNext is not on PyPI, and its ``colabtools`` dependency is only for Colab, so it
goes in without its dependencies, which are listed here instead, in the versions the demo's runs were
made with (JAX and Haiku break each other's newer releases now and then). From the repository's root::

    uv venv --python 3.12 .venv-weathernext && source .venv-weathernext/bin/activate
    uv pip install -e .  # probly
    uv pip install "jax==0.12.0.1"  # on a CPU; on an NVIDIA GPU: "jax[cuda12]==0.12.0.1"
    uv pip install chex==0.1.92 dask==2026.8.0 dinosaur-dycore==1.2.1 dm-haiku==0.0.17 dm-tree==0.1.10 \
        fiddle==0.3.1 h5netcdf==1.8.2 h5py==3.16.0 jraph==0.0.6.dev0 matplotlib==3.11.2 \
        netcdf4==1.7.5.1.post1 numpy==2.5.4 pandas==3.1.0 pyproj==3.8.0 rtree==1.4.1 scipy==1.18.1 \
        trimesh==5.2.0 typing_extensions==4.16.0 xarray==2026.2.0 xarray_tensorstore==0.3.0 \
        "gdm-xarray-jax @ git+https://github.com/google-deepmind/xarray_jax.git@v0.1.1"
    git clone https://github.com/google-deepmind/weathernext ../weathernext
    git -C ../weathernext checkout f2f2c51
    uv pip install --no-deps -e ../weathernext

Without uv, ``python -m venv`` and ``pip install`` do the same.

Run
---
From the repository's root, first a check (one step of one run: the downloads, then a minute or two),
then the runs::

    python docs/source/demos/milton/compute/run_mini.py --check
    python docs/source/demos/milton/compute/run_mini.py

On a CPU, a run takes about 12 minutes on four cores and 11 GB of memory, so the 16 runs take about
3 hours; on a GPU, they take minutes. The script prints every step with the time. When it is done,
commit ``docs/source/demos/milton/assets/data.js``. To look at the demo without building the docs,
``python docs/source/demos/milton/compute/preview.py`` writes ``compute/cache/preview.html``.
"""

from __future__ import annotations

from forecast import run

if __name__ == "__main__":
    run(
        "mini",
        config="WeatherNextCyclones_Mini",
        data="dataset/source-hres_forecast_init-2024-10-07 00:00:00_res-1.0_levels-13_steps-20.nc",
        checkpoints={
            "mini_2024": "params/WeatherNextCyclones_Mini_<2024.npz",  # trained on data through 2023
            "mini_2023": "params/WeatherNextCyclones_Mini_<2023.npz",  # trained on data through 2022
        },
        description=__doc__,
    )
