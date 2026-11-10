"""Leave-one-condition/fluence/speed-out validation of the forward GPRs."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut
from threadpoolctl import threadpool_limits

from data_preparation import MODALITIES, load_data
from forward_gpr import curve_errors, forward_model, loss_weights


def main():
    x, y, repeats = load_data()
    splits = dict(condition=LeaveOneOut().split(x),
                  fluence=LeaveOneGroupOut().split(x, groups=x[:, 0]),
                  speed=LeaveOneGroupOut().split(x, groups=x[:, 1]))
    splits = {name: list(folds) for name, folds in splits.items()}
    rows = []
    for m in MODALITIES:
        for name, folds in splits.items():
            gp_error, mean_error = np.empty(len(x)), np.empty(len(x))
            for train, test in folds:
                weights = loss_weights(m, x[train], y[m][train], repeats)
                pred = forward_model(m).fit(x[train], y[m][train]).predict(x[test])
                mean = np.tile(y[m][train].mean(axis=0), (len(test), 1))
                gp_error[test] = curve_errors(y[m][test], pred, weights)
                mean_error[test] = curve_errors(y[m][test], mean, weights)
            gp, baseline = np.sqrt(np.mean(gp_error ** 2)), np.sqrt(np.mean(mean_error ** 2))
            rows.append(dict(split=name, modality=m, gp_rmse=gp, mean_rmse=baseline, rmse_ratio=gp / baseline))
            print(rows[-1], flush=True)
    Path("results").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv("results/forward_gpr.csv", index=False)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
