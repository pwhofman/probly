"""=====================================================
Mixing torch-uncertainty and probly
=====================================================

Use a torch-uncertainty MC Dropout model, calibrate its averaged logits with
probly temperature scaling, then wrap the calibrated predictor with
torch-uncertainty conformal prediction.
"""

from __future__ import annotations

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch_uncertainty.models.wrappers import mc_dropout
from torch_uncertainty.post_processing import ConformalClsTHR

from probly.calibrator import calibrate
from probly.evaluation import coverage, efficiency
from probly.method.calibration import vector_scaling
from probly.representer import representer

torch.manual_seed(0)
ALPHA = 0.1
NUM_ESTIMATORS = 8

# %%
# Data
# ----

X, y = load_iris(return_X_y=True)
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.6, stratify=y, random_state=0)
X_temp, X_tmp, y_temp, y_tmp = train_test_split(X_tmp, y_tmp, test_size=0.67, stratify=y_tmp, random_state=1)
X_cp, X_test, y_cp, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=2)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_temp = torch.tensor(X_temp, dtype=torch.float32)
X_cp = torch.tensor(X_cp, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_temp = torch.tensor(y_temp, dtype=torch.long)
y_cp = torch.tensor(y_cp, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# %%
# Train a small dropout classifier
# --------------------------------

model = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(16, 3),
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
loss_fn = nn.CrossEntropyLoss()

model.train()
for _ in range(250):
    optimizer.zero_grad()
    loss_fn(model(X_train), y_train).backward()
    optimizer.step()

# %%
# torch-uncertainty: MC Dropout
# -----------------------------

tu_model = mc_dropout(model, num_estimators=NUM_ESTIMATORS, task="classification")
tu_model.eval()


class MeanEstimators(nn.Module):
    """Average TU's concatenated estimator outputs back to one logit row per input."""

    def __init__(self, num_estimators: int) -> None:
        super().__init__()
        self.num_estimators = num_estimators

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.reshape(self.num_estimators, -1, logits.shape[-1]).mean(0)


# %%
# probly: vector scaling
# ---------------------------
# Put a tiny averaging layer after TU's MC Dropout wrapper so downstream
# calibration and conformal prediction see ordinary ``[batch, classes]`` logits.

mean_tu_model = nn.Sequential(tu_model, MeanEstimators(NUM_ESTIMATORS))
temp_model = vector_scaling(mean_tu_model, predictor_type="logit_classifier")
calibrate(temp_model, y_temp, X_temp)

# %%
# torch-uncertainty: conformal prediction on the calibrated model
# ---------------------------------------------------------------

cp_model = ConformalClsTHR(alpha=ALPHA, model=temp_model, enable_ts=False)
cp_loader = DataLoader(TensorDataset(X_cp, y_cp), batch_size=32)
calibrate(cp_model, cp_loader)

# %%
# probly: use the final conformal predictor as a probly representation
# --------------------------------------------------------------------

prediction_sets = representer(cp_model)(X_test)

print(f"coverage: {coverage(prediction_sets, y_test):.3f}")
print(f"average set size: {efficiency(prediction_sets):.3f}")
