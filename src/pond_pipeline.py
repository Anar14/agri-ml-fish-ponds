"""Shared loading, cleaning and modelling helpers for the IoT fish pond thesis.

Every notebook imports from here, so the thresholds, units and date handling
are defined once and stay consistent across the whole analysis.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

RANDOM_STATE = 42

# Bump when the cleaning changes. Notebook 03 writes it next to the processed files,
# and notebooks 04-06 warn if those files were made by an older version.
PIPELINE_VERSION = "2026-11-25.4 (dates per column, stale weights, pond 11 restock, DO 0.01-20)"
VERSION_FILE = PROCESSED_DIR / "pipeline_version.txt"

# ---------------------------------------------------------------------------
# Column names (units are mg/L; some raw files label them g/ml by mistake)
# ---------------------------------------------------------------------------
TEMP = "Temperature (C)"
TURB = "Turbidity(NTU)"
DO = "Dissolved Oxygen(mg/L)"
PH = "PH"
AMMONIA = "Ammonia(mg/L)"
NITRATE = "Nitrate(mg/L)"
POPULATION = "Population"
LENGTH = "Fish_Length(cm)"
WEIGHT = "Fish_Weight(g)"

SENSOR_COLS = [TEMP, TURB, DO, PH, AMMONIA, NITRATE]
# Turbidity and ammonia are dropped before modelling: 65 % and 76 % of their
# readings fall outside the valid range (see notebook 02).
MODEL_SENSOR_COLS = [TEMP, DO, PH, NITRATE]

# ---------------------------------------------------------------------------
# Reference ranges for catfish aquaculture (thesis, section on invalid values).
# Used to REPORT how many readings fall outside them (notebook 01, thesis Table 3).
# (low, high); None means no bound on that side.
# ---------------------------------------------------------------------------
VALID_RANGES: dict[str, tuple[float | None, float | None]] = {
    TEMP: (20.0, 30.0),
    PH: (6.5, 8.5),
    DO: (5.0, None),  # below 5 mg/L stresses catfish: a welfare threshold, reported but not removed
    TURB: (0.0, 50.0),
    NITRATE: (0.0, 25.0),
    AMMONIA: (0.0, 0.1),
}

# Ranges used to REMOVE readings as sensor faults before imputation (notebook 03).
# Same as above, except dissolved oxygen: low oxygen is a real pond condition,
# so only impossible values are removed: exactly 0 or below (the sensor reports
# 0 when it drops out; a pond at 0 mg/L for hours would have no living fish)
# and above 20 mg/L (far above saturation at these temperatures, about 8 mg/L).
CLEANING_RANGES: dict[str, tuple[float | None, float | None]] = {
    **VALID_RANGES,
    DO: (0.01, 20.0),
}

# Ponds whose Population column is empty in the raw export. The values are the
# stocking numbers recorded for these ponds in the thesis. Verify before reuse.
POPULATION_OVERRIDES = {7: 50, 8: 75, 9: 50}

# A new batch of fish was stocked in pond 11 around 4 Feb 2022 (first weight 303 g after
# 701 g in December). Only a few days of the new batch were logged, too few to model, so
# rows from this date on are left out instead of being mixed into the first batch's growth.
RESTOCK_DATES = {11: "2022-02-04"}

_TZ_SUFFIX = re.compile(r"\s+(?!AM$|PM$)[A-Z]{2,4}$")  # "CET", "UTC"; not AM/PM


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=lambda c: c.strip().replace("(g/ml)", "(mg/L)"))
    return df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])


def _date_formats(order: str) -> list[str]:
    """Slash, dash and dot dates in one day/month order, with 4- or 2-digit years and 24h or AM/PM times."""
    out = []
    for sep in "/-.":
        for year in ("%Y", "%y"):
            date = sep.join(["%d", "%m", year] if order == "dm" else ["%m", "%d", year])
            out += [f"{date} %H:%M:%S", f"{date} %H:%M", f"{date} %I:%M:%S %p", f"{date} %I:%M %p", date]
    return out


_SLASH_FORMATS = {True: _date_formats("dm"), False: _date_formats("md")}
_ISO_FORMATS = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S.%f",
                "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M", "%Y-%m-%d", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M", "%Y/%m/%d"]


def parse_timestamps(values: pd.Series, dayfirst: bool = True) -> pd.Series:
    """Parse the mixed timestamp formats found in the raw files.

    ISO rows (``2021-06-25 15:48:59 CET``) are unambiguous. Slash or dash dates
    (``7/11/2021``) are parsed with ONE fixed day/month order for the whole
    column. Letting pandas guess per row (``format="mixed"``) swaps day and
    month only where the day is 12 or less, which scrambles the time order;
    ``load_pond`` picks the order that keeps the file chronological.
    """
    s = values.astype("string").str.strip().str.replace(_TZ_SUFFIX, "", regex=True)
    # some rows carry a UTC offset such as "+01:00"; drop it so every timestamp is local, tz-naive time
    s = s.str.replace(r"\s*(?:[+-]\d{2}:?\d{2}|Z)$", "", regex=True)
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    for fmt in [*_ISO_FORMATS, *_SLASH_FORMATS[dayfirst]]:
        rest = out.isna() & s.notna()
        if not rest.any():
            break
        out.loc[rest] = pd.to_datetime(s[rest], format=fmt, errors="coerce")
    return out


def _chronology(ts: pd.Series) -> float:
    """Share of consecutive rows (in file order) whose timestamp does not go backwards."""
    v = ts.dropna()
    return float((v.diff().dropna() >= pd.Timedelta(0)).mean()) if len(v) > 1 else 0.0


def choose_timestamps(df: pd.DataFrame) -> tuple[pd.Series, dict]:
    """Try each date column and day/month order; keep the one that is most complete and chronological.

    Rows in each export are logged in time order (``entry_id``), so the right
    reading of an ambiguous date is the one in which time never jumps backwards.
    Gaps in the main column are filled from the other column, which may use
    its own day/month order. ``parsed`` is measured on rows that hold sensor
    data, because trailing empty rows are dropped anyway.
    """
    cols = [c for c in ("created_at", "Date") if c in df.columns]
    has_data = df[[c for c in SENSOR_COLS if c in df.columns]].notna().any(axis=1)
    parsed = {(c, d): parse_timestamps(df[c], d) for c in cols for d in (True, False)}
    candidates = []
    for col in cols:
        others = [c for c in cols if c != col]
        for dayfirst in (True, False):
            for other_dayfirst in ((True, False) if others else (None,)):
                ts = parsed[(col, dayfirst)]
                for other in others:  # fill gaps from the other column
                    ts = ts.fillna(parsed[(other, other_dayfirst)])
                candidates.append({"column": col, "dayfirst": dayfirst, "ts": ts,
                                   "parsed": float(ts[has_data].notna().mean()) if has_data.any() else 0.0,
                                   "chronological": _chronology(ts)})
    best = max(candidates, key=lambda c: (round(c["parsed"], 3), c["chronological"]))
    info = {k: best[k] for k in ("column", "dayfirst", "parsed", "chronological")}
    info["unparsed rows with data"] = int((best["ts"].isna() & has_data).sum())
    return best["ts"], info


def unparsed_timestamps(path: str | Path, n: int = 10) -> pd.DataFrame:
    """Diagnostic: the most common raw date strings that could not be read, for rows that hold sensor data."""
    df = _standardise_columns(pd.read_csv(path, low_memory=False))
    ts, _ = choose_timestamps(df)
    has_data = df[[c for c in SENSOR_COLS if c in df.columns]].notna().any(axis=1)
    bad = df.loc[ts.isna() & has_data, [c for c in ("created_at", "Date") if c in df.columns]]
    return bad.astype("string").fillna("<empty>").value_counts().head(n).rename("rows").reset_index()


def pond_number(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    if not match:
        raise ValueError(f"No pond number in file name: {path.name}")
    return int(match.group(1))


def fix_stale_weights(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Give every row of a day the latest sampling weight, ignoring stale values carried over.

    Fish are weighed on sampling days. In some exports (pond 11 most clearly)
    the sensor rows between sampling days keep an old weight, so the series
    jumps up on each sampling day and falls back the next day. Here each day
    takes the highest weight logged that day, and a day whose weight is lower
    than an earlier sampling weight keeps that earlier weight, because the
    average fish in a pond does not shrink. Length is taken from the same row
    as the weight. Returns the frame and the number of rows whose weight changed.
    """
    d = df.sort_values("timestamp", kind="stable")
    day = d["timestamp"].dt.normalize()
    best_row = d[WEIGHT].groupby(day).idxmax().dropna()          # row holding each day's highest weight
    daily = d.loc[best_row.values, [WEIGHT, LENGTH]].set_index(best_row.index)
    running = daily[WEIGHT].cummax()
    src = daily[WEIGHT].eq(running).cumsum()                     # last day that set the running maximum
    source_day = daily.index.to_series().groupby(src.values).transform("first")
    fixed = daily.loc[source_day.values].set_index(daily.index)
    new_w, new_l = day.map(fixed[WEIGHT]), day.map(fixed[LENGTH])
    changed = int((new_w.notna() & ~np.isclose(new_w, d[WEIGHT], equal_nan=True)).sum())
    out = d.copy()
    out[WEIGHT] = new_w.fillna(d[WEIGHT])
    out[LENGTH] = new_l.where(new_w.notna(), d[LENGTH])
    return out.loc[df.index], changed


def load_pond(path: str | Path, fix_weights: bool = True) -> pd.DataFrame:
    """Load one raw pond CSV into a tidy frame with consistent names and units.

    With ``fix_weights`` (the default) stale weights between sampling days are
    replaced, see ``fix_stale_weights``; pass False to see the weights as logged.
    """
    path = Path(path)
    pond = pond_number(path)
    df = _standardise_columns(pd.read_csv(path, low_memory=False))
    if "entry_id" in df.columns:  # file order = logging order
        df = df.sort_values("entry_id", kind="stable", na_position="last").reset_index(drop=True)

    df["timestamp"], date_info = choose_timestamps(df)

    for col in SENSOR_COLS + [POPULATION, LENGTH, WEIGHT]:
        if col not in df.columns:
            df[col] = np.nan
        else:
            # e.g. "Jan-00" in pond 4's length column is an Excel artefact -> NaN
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df[SENSOR_COLS] = df[SENSOR_COLS].replace([np.inf, -np.inf], np.nan)

    # Some exports store nitrate x100 (e.g. 193 instead of 1.93 mg/L).
    if df[NITRATE].median() > VALID_RANGES[NITRATE][1]:
        df[NITRATE] = df[NITRATE] / 100

    if pond in POPULATION_OVERRIDES:
        df[POPULATION] = POPULATION_OVERRIDES[pond]

    df["pond_id"] = pond
    keep = ["pond_id", "timestamp", *SENSOR_COLS, POPULATION, LENGTH, WEIGHT]
    df = df[keep]
    # Rows with no sensor reading at all (trailing rows in some exports) carry
    # no information and would otherwise be filled entirely by the imputer.
    df = df.dropna(subset=SENSOR_COLS, how="all")
    df = df.dropna(subset=["timestamp", WEIGHT])
    df = df.sort_values("timestamp").reset_index(drop=True)
    restock_rows = 0
    if pond in RESTOCK_DATES:
        after = df["timestamp"] >= pd.Timestamp(RESTOCK_DATES[pond])
        restock_rows = int(after.sum())
        df = df[~after].reset_index(drop=True)
    changed = 0
    if fix_weights:
        df, changed = fix_stale_weights(df)
    df.attrs["dates"] = date_info
    df.attrs["weights changed"] = changed
    df.attrs["rows after restock dropped"] = restock_rows
    return df


def timestamp_report(folder: str | Path = RAW_DIR) -> pd.DataFrame:
    """Per pond: which date column and day/month order were used, and the resulting time span."""
    rows = []
    for f in sorted(Path(folder).glob("*.csv"), key=pond_number):
        d = load_pond(f)
        info = d.attrs.get("dates", {})
        rows.append({"pond_id": pond_number(f), **info,
                     "start": d["timestamp"].min(), "end": d["timestamp"].max(),
                     "days": (d["timestamp"].max() - d["timestamp"].min()).days, "rows": len(d),
                     "stale weights replaced": d.attrs.get("weights changed", 0),
                     "rows after restock dropped": d.attrs.get("rows after restock dropped", 0)})
    return pd.DataFrame(rows).set_index("pond_id")


def weight_by_day(df: pd.DataFrame, pond: int) -> pd.DataFrame:
    """Diagnostic: rows and distinct fish weights recorded on each day of one pond."""
    d = df[df["pond_id"] == pond]
    g = d.groupby(d["timestamp"].dt.date)[WEIGHT]
    return pd.DataFrame({"rows": g.size(), "distinct weights": g.nunique(),
                         "min weight": g.min(), "max weight": g.max()})


def load_all_ponds(folder: str | Path = RAW_DIR, fix_weights: bool = True) -> pd.DataFrame:
    files = sorted(Path(folder).glob("*.csv"), key=pond_number)
    if not files:
        raise FileNotFoundError(
            f"No pond CSVs in {folder}. See data/README.md for how to get the raw data."
        )
    return pd.concat([load_pond(f, fix_weights) for f in files], ignore_index=True)


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def out_of_range_mask(df: pd.DataFrame, ranges=VALID_RANGES) -> pd.DataFrame:
    """Boolean frame: True where a reading is outside its range."""
    mask = pd.DataFrame(False, index=df.index, columns=list(ranges))
    for col, (low, high) in ranges.items():
        if low is not None:
            mask[col] |= df[col] < low
        if high is not None:
            mask[col] |= df[col] > high
    return mask


def wrong_value_summary(df: pd.DataFrame, by: str | None = "pond_id") -> pd.DataFrame:
    """Count and percentage of readings outside the reference ranges (VALID_RANGES), overall or per pond."""
    mask = out_of_range_mask(df)
    if by is None:
        counts = mask.sum().to_frame("Wrong values")
        counts["Percentage (%)"] = 100 * counts["Wrong values"] / len(df)
        return counts
    grouped = mask.groupby(df[by])
    counts = grouped.sum()
    pct = (100 * grouped.mean()).add_suffix(" (%)")
    out = counts.join(pct)
    out.insert(0, "Total rows", df.groupby(by).size())
    return out


def remove_invalid(df: pd.DataFrame, cols=SENSOR_COLS, ranges=CLEANING_RANGES) -> pd.DataFrame:
    """Set readings outside the cleaning ranges to NaN (treated as sensor faults)."""
    df = df.copy()
    mask = out_of_range_mask(df, ranges)
    for col in cols:
        df.loc[mask[col], col] = np.nan
    return df


def remove_iqr_outliers(df: pd.DataFrame, cols, k: float = 1.5) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        df.loc[(df[col] < q1 - k * iqr) | (df[col] > q3 + k * iqr), col] = np.nan
    return df


def knn_impute_scaled(df: pd.DataFrame, cols, n_neighbors: int = 5) -> pd.DataFrame:
    """KNN imputation on standardised columns (so no column dominates the distance).

    Only sensor columns are used: the target (fish weight) and fish length are
    never seen by the imputer, so no target information leaks into features.
    """
    df = df.copy()
    usable = [c for c in cols if df[c].notna().any()]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[usable])
    filled = KNNImputer(n_neighbors=n_neighbors).fit_transform(scaled)
    df[usable] = scaler.inverse_transform(filled)
    return df


def polynomial_impute(df: pd.DataFrame, cols, degree: int = 2, clip: bool = True) -> pd.DataFrame:
    """Fill each sensor column from the other sensor columns with a polynomial regression.

    For each column, a degree-2 polynomial regression is fitted on the rows where
    that column is observed, using the other sensor columns as inputs (their own
    gaps are mean-filled first, so every row can be predicted). Only sensor
    columns are used: fish weight and length never enter the imputer.
    With ``clip=True`` imputed values are kept within the range observed in that
    pond (which already lies inside the valid range), because a polynomial can
    extrapolate far beyond the data.
    """
    df = df.copy()
    usable = [c for c in cols if df[c].notna().any()]
    data = df[usable]
    base = pd.DataFrame(SimpleImputer().fit_transform(data), index=data.index, columns=usable)
    for col in usable:
        missing = data[col].isna()
        others = [c for c in usable if c != col]
        if not missing.any():
            continue
        if not others:  # nothing to regress on: fall back to the mean
            df.loc[missing, col] = base.loc[missing, col]
            continue
        poly = PolynomialFeatures(degree)
        observed = ~missing
        model = LinearRegression().fit(poly.fit_transform(base.loc[observed, others]), data.loc[observed, col])
        pred = model.predict(poly.transform(base.loc[missing, others]))
        if clip:  # a polynomial can extrapolate far outside the data; stay within observed values
            pred = np.clip(pred, data.loc[observed, col].min(), data.loc[observed, col].max())
        df.loc[missing, col] = pred
    return df


IMPUTERS = {"polynomial": polynomial_impute, "knn": knn_impute_scaled}


def clean_pond(df: pd.DataFrame, cols=MODEL_SENSOR_COLS, iqr: bool = True,
               method: str = "polynomial") -> pd.DataFrame:
    """Range check -> optional IQR outlier removal -> imputation per pond.

    ``method="polynomial"`` (default) had the lowest error in notebook 02;
    ``method="knn"`` reproduces the imputation used in the thesis.
    """
    df = remove_invalid(df, cols)
    if iqr:
        df = remove_iqr_outliers(df, cols)
    return IMPUTERS[method](df, cols)


# ---------------------------------------------------------------------------
# Modelling table
# ---------------------------------------------------------------------------
def pond_day_table(df: pd.DataFrame, sensor_cols=MODEL_SENSOR_COLS) -> pd.DataFrame:
    """Aggregate 20-second readings to one row per pond per day.

    Fish weight is measured on sampling days and repeated on every sensor row
    in between, so row-level modelling multiplies the same target thousands of
    times. One row per pond-day keeps each observation independent enough to
    evaluate honestly.
    """
    d = df.assign(date=df["timestamp"].dt.normalize())
    agg = {c: "mean" for c in sensor_cols}
    agg.update({POPULATION: "first", WEIGHT: "mean"})
    out = d.groupby(["pond_id", "date"], as_index=False).agg(agg)
    start = out.groupby("pond_id")["date"].transform("min")
    out["days_since_start"] = (out["date"] - start).dt.days
    return out


def write_pipeline_version() -> None:
    VERSION_FILE.write_text(PIPELINE_VERSION, encoding="utf-8")


def check_processed(path: str | Path, date_col: str = "date") -> pd.DataFrame:
    """Print which processed file is used, when it was written and by which pipeline version.

    Warns if the file was made by older code, for example when notebook 03 ran in a
    kernel that had imported an earlier ``src/pond_pipeline.py``. Returns rows and
    date span per pond, so a re-run can be compared with the previous one.
    """
    import datetime as _dt
    path = Path(path)
    made_by = VERSION_FILE.read_text(encoding="utf-8").strip() if VERSION_FILE.exists() else "unknown (written before versioning)"
    print(f"file:     {path}")
    print(f"written:  {_dt.datetime.fromtimestamp(path.stat().st_mtime):%Y-%m-%d %H:%M:%S}")
    print(f"made by:  {made_by}")
    print(f"src code: {PIPELINE_VERSION}  ({Path(__file__).resolve()})")
    if made_by != PIPELINE_VERSION:
        print("WARNING: the processed files were made by different code. Restart the kernel of "
              "notebook 03 (Kernel > Restart & Run All), then re-run this notebook.")
    d = pd.read_csv(path, parse_dates=[date_col])
    return d.groupby("pond_id").agg(rows=(date_col, "size"), first=(date_col, "min"),
                                    last=(date_col, "max"), max_weight=(WEIGHT, "max"))


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class SafeXGBClassifier(ClassifierMixin, BaseEstimator):
    """XGBClassifier that tolerates a class missing from a training fold.

    XGBoost needs labels 0..k-1. With group (pond-wise) splits a fold may lack a
    class, so labels are re-encoded per fit and ``classes_`` keeps the originals,
    which lets ``cross_val_predict(..., method="predict_proba")`` align columns.
    """

    def __init__(self, **params):
        self.params = params

    def get_params(self, deep=True):
        return dict(self.params)

    def set_params(self, **params):
        self.params.update(params)
        return self

    def fit(self, X, y):
        from xgboost import XGBClassifier

        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        self.model_ = XGBClassifier(**self.params)
        self.model_.fit(X, self._le.transform(y))
        return self

    def predict_proba(self, X):
        proba = self.model_.predict_proba(X)
        if proba.ndim == 1:  # single class seen during fit
            proba = proba.reshape(-1, 1)
        return proba

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
