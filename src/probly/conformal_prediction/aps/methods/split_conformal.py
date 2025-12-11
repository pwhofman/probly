"""Split Conformal Prediction Method Implementation."""

from __future__ import annotations

from typing import NotRequired, TypedDict, cast

import numpy as np


class SplitInfo(TypedDict):
    """Type definition for split information dictionary."""

    train_indices: NotRequired[np.ndarray]
    cal_indices: NotRequired[np.ndarray]
    n_training: NotRequired[int]
    n_calibration: NotRequired[int]
    calibration_ratio_used: NotRequired[float]
    calibration_ratio_actual: NotRequired[float]
    used_default: NotRequired[bool]
    default_ratio: NotRequired[float]


class SplitConformal:
    """Implementiert die Split-Conformal Methode.

    This class splits data into training and calibration sets
    for conformal prediction using the split conformal approach.
    The split is done randomly based on a specified calibration ratio.
    """

    def __init__(
        self,
        calibration_ratio: float = 0.3,
        random_state: int | None = None,
    ) -> None:
        """Initialize the SplitConformal class.

        Args:
            calibration_ratio: ratio of data to use for calibration (default: 0.3).
            random_state: for reproducibility of random splits.
        """
        self.calibration_ratio = calibration_ratio
        self.random_state = random_state

        # modern random generator
        self.rng = np.random.default_rng(random_state)

        # save info about last split
        self.last_split_info: SplitInfo | None = None

    def split(
        self,
        x: np.ndarray,
        y: np.ndarray,
        calibration_ratio: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Splits data into training and calibration sets.

        Args:
            x: Features
            y: Labels
            calibration_ratio: overrides default if provided

        Returns:
            x_train, y_train, x_cal, y_cal
        """
        # decide which calibration ratio to use
        if calibration_ratio is not None:
            ratio_to_use = calibration_ratio
            used_default = False
        else:
            ratio_to_use = self.calibration_ratio
            used_default = True

        # validate calibration ratio
        if not 0 < ratio_to_use < 1:
            msg = f"calibration_ratio must be between 0 and 1 (exclusive), got {ratio_to_use}"
            raise ValueError(msg)

        # check if there are at least 2 samples
        if len(x) < 2:
            msg = f"Need at least 2 samples, got {len(x)}"
            raise ValueError(msg)

        # check if x and y have same length
        if len(x) != len(y):
            msg = f"x and y must have same length. Got x:{len(x)}, y:{len(y)}"
            raise ValueError(msg)

        # make sure inputs are numpy arrays
        x = np.asarray(x)
        y = np.asarray(y)

        n_samples = len(x)

        # create shuffled indices
        indices = np.arange(n_samples)
        shuffled_indices = self.rng.permutation(indices)

        # calculate split index
        split_idx = int(n_samples * (1 - ratio_to_use))

        # split indices
        train_indices = shuffled_indices[:split_idx]
        cal_indices = shuffled_indices[split_idx:]

        self.last_split_info = {
            "train_indices": train_indices,
            "cal_indices": cal_indices,
            "n_training": len(train_indices),
            "n_calibration": len(cal_indices),
            "calibration_ratio_used": ratio_to_use,
            "calibration_ratio_actual": len(cal_indices) / n_samples,
            "used_default": used_default,
            "default_ratio": self.calibration_ratio,
        }

        return (
            x[train_indices],
            y[train_indices],
            x[cal_indices],
            y[cal_indices],
        )

    def get_split_info(self) -> SplitInfo | dict[str, str]:
        """Gives information about the last split."""
        if self.last_split_info is None:
            return {"status": "no split performed yet"}
        return self.last_split_info

    def __str__(self) -> str:
        """String representation of the class."""
        info = self.get_split_info()

        if "status" in info:
            # info ist dict[str, str]
            return f"SplitConformal(ratio={self.calibration_ratio}, random_state={self.random_state})"

        # info is SplitInfo
        split_info = cast("SplitInfo", info)
        source = "default" if split_info["used_default"] else "custom"

        return (
            f"SplitConformal: {split_info['n_training']} Training, "
            f"{split_info['n_calibration']} Calibration "
            f"(ratio={split_info['calibration_ratio_used']}, {source})"
        )
