"""GPR forward models and condition-level errors."""
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


def forward_model(modality):
    """Anisotropic Matern 3/2 GPR; Raman uses 8 PCA components."""
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=[1.0, 1.0], length_scale_bounds=(1e-2, 1e2), nu=1.5) + WhiteKernel(1e-2, (1e-8, 1e1))
    gp = make_pipeline(StandardScaler(), GaussianProcessRegressor(
        kernel=kernel, alpha=1e-10, normalize_y=True, n_restarts_optimizer=10, random_state=0))
    transformer = {"raman": PCA(n_components=8, svd_solver="full"),
                   "uvvis": FunctionTransformer(validate=True, check_inverse=False)}.get(modality, StandardScaler())
    return TransformedTargetRegressor(regressor=gp, transformer=transformer, check_inverse=False)


def curve_errors(true, pred, weights=None):
    """Per-condition weighted RMSE over response components."""
    w = 1.0 if weights is None else np.asarray(weights)
    return np.sqrt(np.mean(w * (true - pred) ** 2, axis=1))


def loss_weights(modality, x, y, repeats):
    """Raman: between/within-condition variance ratio (mean 1). Sa/Sz: 1/variance. Training rows only."""
    if modality == "raman":
        spectra, conditions = repeats
        within = np.mean([spectra[(conditions == key).all(axis=1)].var(axis=0, ddof=1) for key in x], axis=0)
        snr = y.var(axis=0, ddof=1) / np.maximum(within, 1e-12)
        return snr / snr.mean()
    if modality == "roughness":
        sd = y.std(axis=0, ddof=1)
        sd[sd < 1e-12] = 1
        return 1 / sd ** 2
    return None
