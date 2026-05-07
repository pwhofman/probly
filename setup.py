"""Setuptools build customization for probly."""

from __future__ import annotations

from pathlib import Path
import shutil

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildPyWithStubs(build_py):
    """Copy external type stubs into setuptools' package build output."""

    def run(self) -> None:
        """Build Python modules, then overlay bundled stubs for the wheel."""
        super().run()

        root = Path(__file__).parent
        stubs_dir = root / "stubs" / "probly"
        package_build_dir = Path(self.build_lib) / "probly"

        if not stubs_dir.exists():
            return

        shutil.copytree(stubs_dir, package_build_dir, dirs_exist_ok=True)
        (package_build_dir / "py.typed").touch()


setup(cmdclass={"build_py": BuildPyWithStubs})
