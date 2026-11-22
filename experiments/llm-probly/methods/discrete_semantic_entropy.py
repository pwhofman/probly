import time
from methods._se_common import compute_se_and_dse

NAME = "discrete_semantic_entropy"
FAMILY = "agreement"
NEEDS_MODEL = False


def score(rows, model=None, tok=None):
    ids, unc, failed = [], [], []
    t0 = time.time()
    for i, r in enumerate(rows):
        ids.append(r.get("id", i))
        try:
            _, dse = compute_se_and_dse(r)
            unc.append(dse)
            failed.append(False)
        except Exception:
            unc.append(1.0)
            failed.append(True)
    return {
        "method": NAME,
        "model": rows[0].get("model"),
        "ids": ids,
        "uncertainty": unc,
        "failed": failed,
        "seconds_per_question": (time.time() - t0) / len(rows),
    }
