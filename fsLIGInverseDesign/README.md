# fsLIGInverseDesign

GPR forward models and RF-guided inverse search for femtosecond laser-induced graphene.

## Setup

Python 3.14 (64-bit).

```sh
pip install -r requirements.txt
```

## Data

Measurement data are not included. Place them in `data/`:

```
data/
  uvvis/jig.txt                  jig reflectance
  uvvis/<v>_<P>.txt              UV-Vis reflectance
  raman/<v>_<P>_<replicate>.csv  Raman spectra
  contact_angle.xlsx             sheet "Plot"
  roughness.xlsx                 sheet "All Results"
```

`<v>` is scan speed in mm/s and `<P>` is average power in units of 0.01 W (e.g. `30_034.txt`).

## Code

- `data_preparation.py`: process coordinates and measurement loading
- `forward_gpr.py`: forward GPR models and error metrics
- `rf_inverse.py`: RF-guided inverse search and candidate selection
- `evaluate_forward_gpr.py`: leave-one-condition/fluence/speed-out validation
- `evaluate_rf_inverse.py`: held-out inverse search for the 30 conditions

## Run

```sh
python evaluate_forward_gpr.py
python evaluate_rf_inverse.py
```

Results are written to `results/`.
