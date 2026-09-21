"""=============================================
Uncertainty for Hugging Face transformers
=============================================

Hugging Face transformers models plug directly into probly: the transformers
integration unwraps their ``ModelOutput`` into logits, so methods like MC
dropout and deep ensembles work on them like on any other torch module.

This example builds a tiny BERT classifier from a config (no downloads), trains
it on a synthetic token classification task, and shows that both MC dropout and
a deep ensemble assign higher uncertainty to out-of-distribution token
sequences.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import torch
from torch import nn
from transformers import BertConfig, BertForSequenceClassification

from probly.method import dropout, ensemble
from probly.quantification import quantify
from probly.representer import representer

torch.manual_seed(0)

# %%
# Data
# ----
#
# Synthetic sequences over a vocabulary of 60 tokens: class 0 draws its tokens
# from one range, class 1 from another. Out-of-distribution sequences use
# tokens that never appear during training.


def sample_sequences(low: int, high: int, n: int, length: int = 12) -> torch.Tensor:
    return torch.randint(low, high, (n, length))


X_train = torch.cat([sample_sequences(5, 30, 200), sample_sequences(30, 55, 200)])
y_train = torch.cat([torch.zeros(200, dtype=torch.long), torch.ones(200, dtype=torch.long)])
X_test = torch.cat([sample_sequences(5, 30, 100), sample_sequences(30, 55, 100)])
y_test = torch.cat([torch.zeros(100, dtype=torch.long), torch.ones(100, dtype=torch.long)])
X_ood = sample_sequences(55, 60, 100)

# %%
# Model
# -----
#
# A tiny BERT classifier built from a config; ``dropout`` inserts MC dropout
# layers exactly as it would for a plain torch model.


def make_model() -> BertForSequenceClassification:
    config = BertConfig(
        vocab_size=60,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=16,
        num_labels=2,
    )
    return BertForSequenceClassification(config)


def train(model: nn.Module, epochs: int = 100) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    for _epoch in range(epochs):
        opt.zero_grad()
        logits = model(X_train).logits
        loss = nn.functional.cross_entropy(logits, y_train)
        loss.backward()
        opt.step()
    model.eval()


# %%
# MC dropout
# ----------

dropout_model = dropout(make_model(), p=0.1, predictor_type="logit_classifier")
train(dropout_model)

rep = representer(dropout_model, num_samples=50)
with torch.no_grad():
    uncertainty_id = quantify(rep.represent(X_test))["total"]
    uncertainty_ood = quantify(rep.represent(X_ood))["total"]

# %%
# Deep ensemble
# -------------
#
# Independently initialized and trained BERT members.

ensemble_model = ensemble(make_model(), num_members=3, predictor_type="logit_classifier")
for member in ensemble_model:
    train(member)

ensemble_rep = representer(ensemble_model)
with torch.no_grad():
    ensemble_uncertainty_id = quantify(ensemble_rep.represent(X_test))["total"]
    ensemble_uncertainty_ood = quantify(ensemble_rep.represent(X_ood))["total"]

# %%
# Uncertainty on in-distribution vs out-of-distribution sequences
# ---------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, name, ood, iid in [
    (axes[0], "MC dropout", uncertainty_ood, uncertainty_id),
    (axes[1], "Deep ensemble", ensemble_uncertainty_ood, ensemble_uncertainty_id),
]:
    ax.hist(iid.numpy(), bins=20, alpha=0.6, label="in-distribution", density=True)
    ax.hist(ood.numpy(), bins=20, alpha=0.6, label="out-of-distribution", density=True)
    ax.set_title(f"{name} total uncertainty")
    ax.set_xlabel("total uncertainty")
    ax.legend()
axes[0].set_ylabel("density")
fig.tight_layout()
plt.show()
