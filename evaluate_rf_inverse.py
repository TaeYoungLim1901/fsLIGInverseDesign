"""Held-out RF-guided inverse search for the 30 measured conditions.

Each target condition is removed from the GPRs, error scales, and RF before its search.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from data_preparation import MODALITIES, load_data
from rf_inverse import fit_fold, prepare_gp, search_target, train_forest


def main():
    x, y, repeats = load_data()
    rows = []
    for held in range(len(x)):
        keep = np.arange(len(x)) != held
        state = prepare_gp(*fit_fold(x[keep], {m: y[m][keep] for m in MODALITIES}, repeats))
        part, evaluations = search_target(state, train_forest(state), {m: y[m][held] for m in MODALITIES}, x[held])
        rows += [dict(target_index=held, **row) for row in part]
        print(f"target {held + 1}/30: {len(part)} returns, {evaluations} GPR evaluations", flush=True)
    table = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    table.to_csv("results/rf_inverse.csv", index=False)
    feasible = table.groupby("target_index").model_feasible.sum()
    print(f"{len(table)} returns, {table.model_feasible.sum()} model-feasible; "
          f"{(feasible >= 1).sum()} targets with >=1 and {(feasible >= 2).sum()} with >=2 feasible returns")


if __name__ == "__main__":
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        main()
