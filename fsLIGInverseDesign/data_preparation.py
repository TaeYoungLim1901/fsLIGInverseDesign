"""Process coordinates and measurement loading."""
import math
from pathlib import Path

import numpy as np
import pandas as pd

MODALITIES = ("uvvis", "raman", "contact", "roughness")

# Area-average fluence F = (P / f) / (pi (d/2)^2), f = 200 kHz, d = 30 um; J/cm^2.
W_TO_F = 1.0 / (200_000.0 * math.pi * (30e-6 / 2.0) ** 2) / 1e4
POWER_W = (0.34, 0.50, 0.75, 1.00, 1.34, 1.60)
SPEED = (30.0, 40.0, 50.0, 60.0, 70.0)
LOW = np.array([POWER_W[0] * W_TO_F, SPEED[0]])
SPAN = np.array([POWER_W[-1] * W_TO_F - POWER_W[0] * W_TO_F, SPEED[-1] - SPEED[0]])

RAMAN_KEEP = {(1.6, 50): {1, 2, 4}, (1.0, 60): {1, 3, 4}}  # replicates selected at measurement
WAVELENGTH = np.arange(300.0, 1402.0, 2.0)
SHIFT = np.arange(200.0, 3200.0, 2.0)


def read_uv(path):
    frame = pd.read_csv(path, skiprows=1).apply(pd.to_numeric, errors="coerce").dropna()
    return frame.iloc[:, 0].to_numpy(float), frame.iloc[:, 1].to_numpy(float)


def load_data(root=Path(__file__).resolve().parent / "data"):
    """Return x (30 x [F, v]), y per property, and Raman repeats (y, x). File layout: README."""
    root = Path(root)
    keys = sorted((p, v) for p in POWER_W for v in SPEED)

    axis, jig = read_uv(root / "uvvis" / "jig.txt")
    assert np.array_equal(axis, WAVELENGTH)
    uv = []
    for p, v in keys:
        wavelength, values = read_uv(root / "uvvis" / f"{v:.0f}_{round(p * 100):03d}.txt")
        assert np.array_equal(wavelength, axis)
        uv.append(values - jig)

    repeats, repeat_keys = [], []
    for path in sorted((root / "raman").glob("*.csv")):
        speed, power, replicate = (int(s) for s in path.stem.split("_"))
        power /= 100
        keep = RAMAN_KEEP.get((power, speed))
        if keep and replicate not in keep:
            continue
        frame = pd.read_csv(path, skiprows=12, header=None).apply(pd.to_numeric, errors="coerce").dropna()
        spectrum = np.interp(SHIFT, frame[0].to_numpy(float), frame[1].to_numpy(float))
        repeats.append(spectrum / np.sqrt(np.mean(spectrum ** 2)))
        repeat_keys.append((power, float(speed)))
    repeats, repeat_keys = np.array(repeats), np.array(repeat_keys)
    groups = [(repeat_keys == key).all(axis=1) for key in keys]
    assert all(g.sum() == 3 for g in groups)
    raman = np.array([repeats[g].mean(axis=0) for g in groups])

    plot = pd.read_excel(root / "contact_angle.xlsx", sheet_name="Plot")
    rough = pd.read_excel(root / "roughness.xlsx", sheet_name="All Results")
    contact, roughness = [], []
    for power, speed in keys:
        hit = plot[np.isclose(plot["Power (W)"], power) & np.isclose(plot["Scannig speed (mm/s)"], speed)]
        contact.append([float(hit["Contact angle Avg. (°)"].iloc[0])])
        row = rough[np.isclose(rough["Power (W)"], power) & np.isclose(rough["Scanning speed (mm/s)"], speed)].iloc[0]
        roughness.append((float(row.Sa_Mean), float(row.Sz_Mean)))

    keys = np.array(keys)
    x = np.column_stack([keys[:, 0] * W_TO_F, keys[:, 1]])
    repeat_keys[:, 0] = repeat_keys[:, 0] * W_TO_F
    y = dict(uvvis=np.array(uv), raman=raman, contact=np.array(contact), roughness=np.array(roughness))
    return x, y, (repeats, repeat_keys)
