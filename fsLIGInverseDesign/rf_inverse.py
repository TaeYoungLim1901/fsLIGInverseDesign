"""RF-guided inverse search over the GPR forward models."""
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from data_preparation import LOW, MODALITIES, SPAN, W_TO_F
from forward_gpr import curve_errors, forward_model, loss_weights

RF_TREES, RF_CENTERS, SYNTHETIC, HALF_WIDTH = 32, 2, 4096, 0.25
RETURN_K, SEPARATION = 5, 0.10


def tensor(array):
    return torch.as_tensor(array, dtype=torch.float64)


def fit_fold(x, y, repeats):
    """Fit the four GPRs and their 3-fold CV error scales on the given conditions."""
    models, calibration = {}, {}
    for m in MODALITIES:
        errors = np.empty(len(x))
        for train, test in KFold(3, shuffle=True, random_state=1).split(x):
            fitted = forward_model(m).fit(x[train], y[m][train])
            errors[test] = curve_errors(y[m][test], fitted.predict(x[test]),
                                        loss_weights(m, x[train], y[m][train], repeats))
        calibration[m] = dict(sigma=float(errors.mean()), weights=loss_weights(m, x, y[m], repeats))
        models[m] = forward_model(m).fit(x, y[m])
    return models, calibration


def prepare_gp(models, calibration):
    """Express each GPR mean as kernel @ coefficients, with an SVD basis in the error metric."""
    blocks = []
    for m in MODALITIES:
        model = models[m]
        scaler, gp = model.regressor_.steps[0][1], model.regressor_.steps[-1][1]
        width = int(gp.alpha_.shape[1])
        offset = model.transformer_.inverse_transform(np.asarray(gp._y_train_mean).reshape(1, width))[0]
        native = model.transformer_.inverse_transform(gp.alpha_ * gp._y_train_std + gp._y_train_mean) - offset
        weights = calibration[m]["weights"]
        n = len(offset)
        scale = np.sqrt(np.ones(n) if weights is None else weights) / np.sqrt(n) / calibration[m]["sigma"]
        left, singular, right = np.linalg.svd(native * scale, full_matrices=False)
        blocks.append(dict(
            m=m, mean=tensor(scaler.mean_), scale=tensor(scaler.scale_), train=tensor(gp.X_train_),
            length=tensor(gp.kernel_.k1.k2.length_scale), amplitude=float(gp.kernel_.k1.k1.constant_value),
            native=tensor(native), b=tensor(offset), metric_scale=tensor(scale),
            v=tensor(right.T), coef=tensor(left * singular)))
    z = np.vstack([np.random.default_rng(765).uniform(size=(50, 2)), [[0, 0], [1, 1]]])
    for block in blocks:
        np.testing.assert_allclose((gp_kernel(tensor(z), block) @ block["native"] + block["b"]).numpy(),
                                   models[block["m"]].predict(LOW + z * SPAN), rtol=1e-8, atol=1e-8)
    return dict(blocks=blocks, sizes=[b["coef"].shape[1] for b in blocks])


def gp_kernel(z, block):
    scaled = (tensor(LOW) + z * tensor(SPAN) - block["mean"]) / block["scale"]
    delta = (scaled[:, None, :] - block["train"][None, :, :]) / block["length"]
    factor = np.sqrt(3) * torch.sqrt(torch.sum(delta * delta, dim=2) + 1e-30)
    return block["amplitude"] * ((1 + factor) * torch.exp(-factor))


def gp_coordinates(state, z):
    return torch.cat([gp_kernel(z, block) @ block["coef"] for block in state["blocks"]], dim=1)


def encode_targets(state, targets):
    coordinates, residual = [], []
    for block in state["blocks"]:
        delta = (tensor(targets[block["m"]]) - block["b"]) * block["metric_scale"]
        coded = delta @ block["v"]
        coordinates.append(coded)
        residual.append(torch.sum((delta - coded @ block["v"].T) ** 2, dim=1))
    return torch.cat(coordinates, 1), torch.stack(residual, 1)


def physical(m, y):
    valid = np.isfinite(y).all(axis=1)
    if m == "roughness":
        return valid & (y >= 0).all(axis=1) & (y[:, 1] >= y[:, 0])
    if m == "contact":
        return valid & (y[:, 0] >= 0) & (y[:, 0] <= 180)
    return valid


def forward_coordinates(state, points):
    coordinates = np.empty((len(points), sum(state["sizes"])))
    valid = np.empty(len(points), bool)
    for start in range(0, len(points), 512):
        stop = min(start + 512, len(points))
        with torch.no_grad():
            block = gp_coordinates(state, tensor(points[start:stop])).numpy()
        coordinates[start:stop] = block
        okay, offset = np.ones(len(block), bool), 0
        for part in state["blocks"]:
            width = part["coef"].shape[1]
            native = block[:, offset:offset + width] @ part["v"].numpy().T
            okay &= physical(part["m"], native / part["metric_scale"].numpy() + part["b"].numpy())
            offset += width
        valid[start:stop] = okay
    return coordinates, valid


def train_forest(state, seed=0):
    """RF from GPR-encoded responses to normalized (F, v), on uniform synthetic points."""
    z = np.random.default_rng(seed).uniform(size=(SYNTHETIC, 2))
    with torch.no_grad():
        coordinates = gp_coordinates(state, tensor(z)).numpy()
    center, spread = coordinates.mean(0), coordinates.std(0).clip(1e-4)
    forest = RandomForestRegressor(n_estimators=RF_TREES, random_state=seed, n_jobs=1)
    forest.fit((coordinates - center) / spread, z)
    return dict(model=forest, center=center, spread=spread)


def propose_centers(trained, query):
    """Cluster the per-tree predictions into search centers, largest cluster first."""
    predictions = np.array([tree.predict(query)[0] for tree in trained["model"].estimators_])
    count = min(RF_CENTERS, len(np.unique(predictions, axis=0)))
    if count == 1:
        return np.clip(predictions.mean(0)[None], 0, 1)
    clusters = KMeans(n_clusters=count, n_init=10, random_state=trained["model"].random_state).fit(predictions)
    order = np.argsort(-np.bincount(clusters.labels_, minlength=count), kind="stable")
    return np.clip(clusters.cluster_centers_[order], 0, 1)


def search_lattice(centers):
    """Controller points (0.01 W, 0.01 mm/s steps) in the union of the clipped windows."""
    ids = []
    for center in centers:
        low = LOW + SPAN * np.maximum(0, center - HALF_WIDTH)
        high = LOW + SPAN * np.minimum(1, center + HALF_WIDTH)
        power, speed = np.meshgrid(
            np.arange(int(np.ceil(low[0] / W_TO_F * 100 - 1e-9)), int(np.floor(high[0] / W_TO_F * 100 + 1e-9)) + 1),
            np.arange(int(np.ceil(low[1] * 100 - 1e-9)), int(np.floor(high[1] * 100 + 1e-9)) + 1), indexing="ij")
        ids.append(((power - 34) * 4001 + speed - 3000).ravel())
    union = np.unique(np.concatenate(ids))
    native = np.column_stack([((34 + union // 4001) / 100) * W_TO_F, (3000 + union % 4001) / 100])
    return (native - LOW) / SPAN


def rank_candidates(points, losses, valid):
    """Feasible (all L <= 1) by mean L, else valid; greedy separation > SEPARATION."""
    feasible = valid & (losses.max(1) <= 1)
    order = np.flatnonzero(feasible if feasible.any() else valid)
    order = order[np.argsort(losses[order].mean(1), kind="stable")]
    chosen = []
    while len(order) and len(chosen) < RETURN_K:
        index = int(order[0])
        chosen.append(index)
        rest = order[1:]
        order = rest[np.linalg.norm(points[rest] - points[index], axis=1) / np.sqrt(2) > SEPARATION]
    return chosen


def search_target(state, trained, target, origin):
    """Return up to five candidates, excluding points within SEPARATION of the origin condition."""
    encoded, residual = encode_targets(state, {m: np.atleast_2d(target[m]) for m in MODALITIES})
    centers = propose_centers(trained, (encoded.numpy() - trained["center"]) / trained["spread"])
    points = search_lattice(centers)
    coordinates, valid = forward_coordinates(state, points)
    coded, remain = encoded.numpy()[0], residual.numpy()[0]
    losses, offset = np.empty((len(points), len(MODALITIES))), 0
    for i, width in enumerate(state["sizes"]):
        delta = coordinates[:, offset:offset + width] - coded[offset:offset + width]
        losses[:, i] = np.sqrt((delta * delta).sum(1) + remain[i])
        offset += width
    origin_z = (origin - LOW) / SPAN
    distance = np.linalg.norm(points - origin_z, axis=1) / np.sqrt(2)
    rows = []
    for rank, i in enumerate(rank_candidates(points, losses, valid & (distance > SEPARATION)), 1):
        native = LOW + SPAN * points[i]
        rows.append(dict(
            target_F=origin[0], target_v=origin[1], rank=rank,
            F=np.round(native[0] / W_TO_F, 2) * W_TO_F, v=np.round(native[1], 2),
            mean_L=losses[i].mean(), max_L=losses[i].max(), model_feasible=bool(losses[i].max() <= 1),
            origin_NEPD=distance[i], **{m + "_L": losses[i, j] for j, m in enumerate(MODALITIES)}))
    return rows, len(points)
