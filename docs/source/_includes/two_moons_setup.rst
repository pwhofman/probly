.. jupyter-execute::
    :hide-code:

    # Shared, hidden setup for the narrative pages: a trained Two Moons classifier
    # plus in- and out-of-distribution inputs, so the blocks below run for real.
    # Seeded, so the printed numbers are reproducible across builds.
    import random

    import numpy as np
    from sklearn.datasets import make_moons
    import torch
    from torch import nn

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    X, y = make_moons(n_samples=500, noise=0.05, random_state=0)
    data_id = torch.from_numpy(X).float()
    labels = torch.from_numpy(y).long()

    # Out-of-distribution inputs: the same two moons, translated off the manifold
    # the model was fitted on.
    data_ood = data_id + 2.5

    net = nn.Sequential(
        nn.Linear(2, 64),
        nn.ReLU(),
        nn.Dropout(p=0.25),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Dropout(p=0.25),
        nn.Linear(64, 2),
    )

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    net.train()
    for _ in range(300):
        optimizer.zero_grad()
        nn.functional.cross_entropy(net(data_id), labels).backward()
        optimizer.step()
    net.eval()

    x = data_id  # the inputs the representation blocks below are built from
