# Processing Agricultural Data with Machine Learning Techniques

Master's thesis project on machine learning for fish growth in IoT-monitored catfish aquaponics ponds. ESP32
sensor units in 11 ponds logged temperature, turbidity, dissolved oxygen, pH, ammonia and nitrate every few
seconds from June 2021 (about 1.1 million readings, from the public
[Sensor-Based Aquaponics Fish Pond Datasets](https://www.kaggle.com/datasets/ogbuokiriblessing/sensor-based-aquaponics-fish-pond-datasets)).
Fish length and weight were recorded on sampling days. This repository cleans the sensor data and asks:

> How well can water quality and time predict fish weight, and fish size class, in a pond the model has never seen?

## Pipeline

| Notebook | What it does |
| --- | --- |
| [`01_data_overview`](notebooks/01_data_overview.ipynb) | Merges the 11 pond exports, descriptive statistics per pond, counts sensor readings outside the valid range for aquaculture |
| [`02_imputation_comparison`](notebooks/02_imputation_comparison.ipynb) | Hides 10 % of known readings and compares mean, KNN, iterative and polynomial-regression imputation on exactly those cells |
| [`03_preprocessing`](notebooks/03_preprocessing.ipynb) | Range check → IQR outlier removal → polynomial-regression imputation per pond; builds a one-row-per-pond-per-day modelling table |
| [`04_regression`](notebooks/04_regression.ipynb) | Predicts fish weight with linear models, decision tree, random forest and XGBoost against two baselines |
| [`05_classification`](notebooks/05_classification.ipynb) | Predicts five fish size classes with logistic regression, decision tree, random forest and XGBoost |
| [`06_visualization`](notebooks/06_visualization.ipynb) | Compares water quality and fish growth across ponds, and shows where sensor faults are |

Shared code (column names, valid ranges, loading, cleaning) lives in [`src/pond_pipeline.py`](src/pond_pipeline.py),
so every notebook uses the same thresholds and units.

## Methodology choices

- **Reference ranges** from the catfish literature reviewed in the thesis (temperature 20–30 °C, pH 6.5–8.5,
  dissolved oxygen ≥ 5 mg/L, nitrate ≤ 25 mg/L, turbidity 0–50 NTU, ammonia ≤ 0.1 mg/L) are used to report faulty
  readings and, except for oxygen, to clean them. Dissolved oxygen below 5 mg/L is a stress level, not a sensor
  fault, so only impossible values (0 mg/L or less, which is what the sensor reports when it drops out, or above 20 mg/L) are removed. Turbidity and ammonia are out of range in
  most readings and are not used for modelling.
- **Dates and weights.** Each pond's dates are read with one day/month order, chosen so the readings stay in time
  order. Fish weight is logged on sampling days; where sensor rows in between keep an old weight (pond 11), each
  day takes the latest sampling weight instead. Pond 11 was restocked with new fish on 4 Feb 2022; the few days
  after that are left out.
- **No target leakage.** Fish length is measured together with weight and maps almost one-to-one to it, so it is
  not a feature. The imputer only sees sensor columns, never the target.
- **Honest evaluation.** Readings are aggregated to one row per pond per day, and models are cross-validated
  with `GroupKFold` by pond, so test ponds are never seen in training. A random split of 20-second readings would
  put near-identical rows in train and test and report near-perfect but meaningless scores.
- **Imputation.** Notebook 02 hides known readings and scores each method on them. Polynomial regression on the
  other sensors had the lowest error (mean NRMSE 0.74, vs 0.79 iterative, 0.98 KNN, 1.00 mean) and is used for
  cleaning. The thesis used KNN; `clean_pond(..., method="knn")` reproduces it.
- **Size classes.** As in the thesis, fish weight is cut into five equal-width classes; equal-frequency classes are
  reported alongside. Models are compared on macro F1 and balanced accuracy, because plain accuracy rewards
  predicting the largest class.
- **Baselines.** Every model is compared with predicting the mean and with a time-only model, to show what the
  water-quality sensors add beyond fish age.

## Results

All scores are out-of-pond: 5-fold `GroupKFold` by pond on the pond-day table (11 ponds, 856 pond-days), so each
pond is predicted by models that never saw it.

**Fish weight (regression, notebook 04)**

| Model | R² | RMSE (g) | MAE (g) |
|---|---:|---:|---:|
| Baseline: days since start only (linear) | **0.890** | **81.8** | 59.9 |
| XGBoost | 0.879 | 85.7 | **48.8** |
| Lasso (α = 0.1) | 0.863 | 91.3 | 62.4 |
| Linear regression | 0.862 | 91.6 | 62.6 |
| Random forest | 0.810 | 107.3 | 48.9 |
| Decision tree (depth 5) | 0.806 | 108.6 | 53.7 |
| Baseline: mean | −0.199 | 269.7 | 215.6 |

**Size class (classification, notebook 05)**

| Model | Equal-width: balanced acc. | Equal-width: macro F1 | Equal-frequency: balanced acc. | Equal-frequency: macro F1 |
|---|---:|---:|---:|---:|
| Logistic regression | **0.604** | **0.586** | 0.743 | 0.742 |
| Random forest | 0.583 | 0.567 | 0.805 | 0.804 |
| XGBoost | 0.545 | 0.547 | **0.828** | **0.828** |
| Decision tree (depth 6) | 0.565 | 0.541 | 0.772 | 0.774 |
| Baseline: most frequent class | 0.200 | 0.153 | 0.054 | 0.033 |

**What this shows**

- **Fish age explains most of the weight.** A straight line on days since stocking explains 89 % of the variance in
  ponds the model has not seen. No model with the water-quality sensors beats it on R² or RMSE; XGBoost has a lower
  MAE but a higher RMSE. In XGBoost, `days_since_start` carries about 96 % of the feature importance.
- **The sensors add no measurable predictive value here.** With 11 ponds, mostly stable water quality and several
  sensors out of range much of the time, this data cannot show an effect of water quality on growth. That is a
  limit of the data, not evidence that water quality does not matter.
- **Earlier results were wrong for fixable reasons.** Before the cleaning fixes, sensors seemed to help
  (R² 0.53 vs 0.41 for time alone). That came from scrambled day/month dates in ponds 7 and 8 and stale weights in
  pond 11, which broke the link between time and weight. The thesis's near-perfect scores came from target leakage
  (see [CHANGES.md](CHANGES.md)).
- **Equal-width classes are very unbalanced.** About 62 % of pond-days fall in the smallest class and the largest class
  has only 17, so accuracy (0.86–0.88) looks better than it is. Balanced accuracy is about 0.55–0.60, and "very
  large" fish are almost always predicted as "large". With equal-frequency classes the models reach 0.74–0.83.

The full tables are written to [`results/`](results/) by the notebooks
(`regression_results.csv`, `classification_results_*.csv`, `imputation_comparison.csv`, `wrong_values_per_pond.csv`).

## Run it

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab                       # run notebooks/01 ... 06 in order
```

The 11 raw pond CSVs are included in `data/raw/` (about 96 MB), so the notebooks run straight after cloning.
See [`data/README.md`](data/README.md) for the source and citation.

## Repository layout

```
├── data/
│   ├── README.md          # source, licence, known data quirks
│   ├── raw/               # the 11 original pond CSVs
│   └── processed/         # created by notebook 03 (not committed)
├── notebooks/             # 01–06, run in order
├── results/               # tables written by the notebooks
├── src/pond_pipeline.py   # shared loading, cleaning and modelling helpers
└── requirements.txt
```

## Changes from the original thesis notebooks

See [CHANGES.md](CHANGES.md) for every change, with old code, new code and the reason.

## Licence

Code: MIT (see [LICENSE](LICENSE)). The data keep the licence of their original source (see [data/README.md](data/README.md)).
