"""
Listeria soil modeling pipeline (clean, annotated .py)

What this script does
---------------------
1) Load the raw Listeria soil dataset
2) Standardize missing values and summarize missingness
3) (Optional) Drop columns with missing values (column-wise deletion) to keep ALL rows
4) Enrich each sampling record with daily weather features from Open-Meteo Archive API
5) Create a binary label (presence/absence) from isolate counts
6) Build a LightGBM binary classifier under an imbalanced, real-world setting
7) Evaluate performance at multiple thresholds (including a lower threshold to reduce FN)
8) Explain model behavior with SHAP summary plot
9) Create a US map of predicted risk, overlaying lakes/rivers, and highlighting high-risk points

Key modeling philosophy
-----------------------
- The goal is NOT to reproduce performance from a more balanced/cleaned dataset.
- The goal is to perform well under more realistic, imbalanced surveillance conditions,
  and extract interpretable signals (SHAP) that support food safety reasoning.

Requirements
------------
pip install pandas numpy scikit-learn lightgbm shap geopandas shapely matplotlib tqdm requests

Notes
-----
- You MUST implement `fetch_openmeteo_hourly_day()` (API call + daily aggregation).
  A stub is included for clarity.
- If you run on Windows and have GeoPandas install issues, use conda-forge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
from tqdm import tqdm

# Modeling
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# Explainability
import shap

# Mapping
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box


# ---------------------------------------------------------------------
# Configuration (edit these paths/values for your project)
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    raw_csv_path: str = "data/raw/ListeriaSoil_raw.csv"
    random_state: int = 143

    # Weather enrichment settings
    timezone: str = "auto"
    weather_model: str = "era5"  # Open-Meteo archive model

    # Modeling settings
    test_size: float = 0.2
    high_risk_threshold: float = 0.60

    # If True: drop columns with ANY missing values (keeps all rows)
    drop_missing_columns: bool = True


CFG = Config()


# ---------------------------------------------------------------------
# Helper: Convert date to ISO format (YYYY-MM-DD)
# ---------------------------------------------------------------------
def to_ymd(date_value: Any) -> Optional[str]:
    """
    Convert a raw date into ISO format.

    Parameters
    ----------
    date_value : Any
        Raw date value (string, datetime, etc.)

    Returns
    -------
    Optional[str]
        "YYYY-MM-DD" or None if parsing fails
    """
    if pd.isna(date_value):
        return None
    try:
        dt = pd.to_datetime(date_value)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None


# ---------------------------------------------------------------------
# Helper: Missingness summary table (counts + %)
# ---------------------------------------------------------------------
def summarize_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame summarizing missingness per column.
    """
    na_counts = df.isna().sum()
    na_percent = (na_counts / len(df)) * 100
    return (
        pd.DataFrame({"Missing_Count": na_counts, "Missing_%": na_percent.round(2)})
        .sort_values("Missing_Count", ascending=False)
    )


# ---------------------------------------------------------------------
# Weather API fetch (STUB)
# ---------------------------------------------------------------------
def fetch_openmeteo_hourly_day(
    *,
    date_: str,
    lat: float,
    lon: float,
    timezone: str = "auto",
    model: str = "era5",
) -> Dict[str, Any]:
    """
    Fetch hourly weather for a single day and summarize to daily features.

    You should implement:
    - A request to Open-Meteo Archive API:
        https://archive-api.open-meteo.com/v1/archive
    - Request hourly variables like:
        temperature_2m, relative_humidity_2m, precipitation, wind_speed_10m
    - Summarize into daily features (example keys expected by this script):
        temp_mean, temp_min, temp_max
        rh_mean, rh_min, rh_max
        ppt_sum_mm, ppt_hours_gt0
        wind_mean, wind_min, wind_max

    Returns
    -------
    Dict[str, Any]
        Daily weather features for merging back into the sample row.
    """
    raise NotImplementedError(
        "Implement Open-Meteo fetch + daily summarization. "
        "Make sure returned keys match expected weather feature names."
    )


# ---------------------------------------------------------------------
# Helper: Evaluate metrics at a chosen probability threshold
# ---------------------------------------------------------------------
def eval_at_threshold(
    y_true: pd.Series,
    y_proba: np.ndarray,
    threshold: float = 0.5,
    name: str = "model",
) -> pd.DataFrame:
    """
    Compute classification metrics at a given threshold.

    Why thresholds matter in food safety:
    - A lower threshold can reduce false negatives (FN), which is often preferred
      when missing a contamination event is costly.
    """
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    out = {
        "Model": name,
        "Threshold": threshold,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "Accuracy": (tp + tn) / (tp + tn + fp + fn),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Sensitivity (Recall)": recall_score(y_true, y_pred),
        "Specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC AUC": roc_auc_score(y_true, y_proba),
    }
    return pd.DataFrame([out])


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    # =============================================================
    # 1) Load raw data
    # =============================================================
    df_r = pd.read_csv(CFG.raw_csv_path)

    print("Preview (head):")
    print(df_r.head())
    print("\nDataset info:")
    print(df_r.info())
    print("\nDimensions:", df_r.shape)

    # Basic dimension printouts (useful for reporting/data QA)
    n_rows, n_cols = df_r.shape
    print(f"\nNo. of Rows: {n_rows}")
    print(f"No. of Columns: {n_cols}")

    # =============================================================
    # 2) Standardize missing values + missingness summary
    # =============================================================
    # Many raw datasets store missing as "-" (string). Convert those to actual NaN.
    df_r = df_r.replace("-", np.nan)

    na_summary = summarize_missingness(df_r)
    print("\nMissingness summary (top 15):")
    print(na_summary.head(15))

    # =============================================================
    # 3) Create working dataset (df_play)
    #    Keep ALL rows, optionally drop columns with missing values
    # =============================================================
    df_play = df_r.copy()

    if CFG.drop_missing_columns:
        cols_with_na = df_play.columns[df_play.isna().any()].tolist()
        print(f"\nDropping {len(cols_with_na)} columns that contain missing values.")
        df_play = df_play.drop(columns=cols_with_na)

    print("\nShape after cleaning strategy:", df_play.shape)

    # =============================================================
    # 4) Date conversion for weather enrichment
    # =============================================================
    # Ensure the column name matches your raw CSV.
    if "Sampling date" not in df_play.columns:
        raise KeyError(
            "Expected column 'Sampling date' not found. "
            "Update the script to match your dataset column name."
        )

    df_play["date_ymd"] = df_play["Sampling date"].apply(to_ymd)
    bad_dates = df_play["date_ymd"].isna().sum()
    if bad_dates > 0:
        print(f"\nWARNING: {bad_dates} rows have invalid dates (date_ymd is NaN).")

    # =============================================================
    # 5) Fetch weather per row (lat, lon, date)
    # =============================================================
    # Required fields for weather query:
    for col in ["Latitude", "Longitude", "date_ymd"]:
        if col not in df_play.columns:
            raise KeyError(f"Expected column '{col}' not found. Fix your dataset/column names.")

    tqdm.pandas()

    rows: List[Dict[str, Any]] = []

    for idx, r in tqdm(df_play.iterrows(), total=len(df_play), desc="Fetching weather"):
        date_ = r["date_ymd"]
        lat = r["Latitude"]
        lon = r["Longitude"]

        # If required fields missing, keep alignment but skip API call
        if pd.isna(date_) or pd.isna(lat) or pd.isna(lon):
            rows.append({"row_index": idx})
            continue

        feats = fetch_openmeteo_hourly_day(
            date_=str(date_),
            lat=float(lat),
            lon=float(lon),
            timezone=CFG.timezone,
            model=CFG.weather_model,
        )

        feats["row_index"] = idx
        rows.append(feats)

    weather_all = pd.DataFrame(rows)

    # Merge daily weather features back to df_play
    df_enriched = (
        df_play.reset_index()
        .merge(weather_all, left_on="index", right_on="row_index", how="left")
        .drop(columns=["row_index"])
    )

    print("\nFinal shape after enrichment:", df_enriched.shape)

    # Optional: check missingness in weather columns (depends on your fetch keys)
    weather_cols = [
        "temp_mean", "temp_min", "temp_max",
        "rh_mean", "rh_min", "rh_max",
        "ppt_sum_mm", "ppt_hours_gt0",
        "wind_mean", "wind_min", "wind_max",
    ]
    existing_weather_cols = [c for c in weather_cols if c in df_enriched.columns]
    if existing_weather_cols:
        print("\nMissingness in weather features (fraction missing):")
        print(df_enriched[existing_weather_cols].isna().mean().sort_values(ascending=False))
    else:
        print("\nNOTE: Expected weather columns not found. Confirm keys returned by fetch_openmeteo_hourly_day().")

    # =============================================================
    # 6) Create binary label (presence/absence)
    # =============================================================
    isolates_col = "Number of Listeria isolates obtained"
    if isolates_col not in df_enriched.columns:
        raise KeyError(
            f"Expected '{isolates_col}' not found. Update column name to match your dataset."
        )

    # label = 1 if isolate count > 0 (presence), else 0 (absence)
    df_enriched["label"] = (pd.to_numeric(df_enriched[isolates_col], errors="coerce") > 0).astype(int)

    # -------------------------------------------------------------
    # Include your requested label checks
    # -------------------------------------------------------------
    y_e = df_enriched["label"]

    print("\nLabel counts (including NaN check just in case):")
    print(y_e.value_counts(dropna=False))
    print("Unique labels:", y_e.unique())

    # =============================================================
    # 7) Define features (drop target + obvious leakage/non-features)
    # =============================================================
    # Drop columns that:
    # - are the target itself ("label")
    # - directly encode the target (isolate counts)
    # - are IDs or metadata not intended as predictors
    # - could leak information (e.g., post-lab selection fields)
    X_e = df_enriched.drop(
        columns=[
            "label",
            "Number of Listeria isolates obtained",
            "Number of Listeria isolates selected for WGS (i.e., number of isolates with unique sigB)",
            "If selected for soil property analysis: Yes(Y)/No(N)",
            "index",
            "Sample ID ",
            "Sampling grid",
            "date_ymd",
            "wx_key",
            "US state",
        ],
        errors="ignore",  # do not crash if a column name is absent
    )

    print("\nShapes -> X:", X_e.shape, " y:", y_e.shape)

    # =============================================================
    # 8) Train/test split (stratified to preserve imbalance ratio)
    # =============================================================
    X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(
        X_e,
        y_e,
        test_size=CFG.test_size,
        stratify=y_e,
        random_state=CFG.random_state,
    )

    print("\nTrain label proportion:")
    print(y_train_e.value_counts(normalize=True))

    print("\nTest label proportion:")
    print(y_test_e.value_counts(normalize=True))

    # Compute a common imbalance weight:
    # pos_weight = (# negatives) / (# positives)
    pos_weight = (len(y_train_e) - y_train_e.sum()) / max(y_train_e.sum(), 1)
    print("\npos_weight (neg/pos):", pos_weight)

    # =============================================================
    # 9) LightGBM model definition + training
    # =============================================================
    # Important knobs for imbalanced learning:
    # - scale_pos_weight: increases penalty for misclassifying positives
    # - class_weight: additional weighting by class (0 and 1)
    # NOTE: You can use either or both; here we keep your setup.
    lgbm = LGBMClassifier(
        objective="binary",
        n_estimators=550,
        learning_rate=0.0009,
        num_leaves=30,
        max_depth=-1,
        min_child_samples=2,
        subsample=0.6,
        colsample_bytree=0.6,
        scale_pos_weight=pos_weight * 2,
        random_state=CFG.random_state,
        n_jobs=-1,
        class_weight={0: 1.95, 1: 4.2},
    )

    # Fit model
    lgbm.fit(X_train_e, y_train_e)

    # Predict probabilities on test set (class 1 probability = predicted risk)
    y_proba = lgbm.predict_proba(X_test_e)[:, 1]

    # ROC AUC is threshold-independent (useful summary metric for imbalanced data)
    auc = roc_auc_score(y_test_e, y_proba)
    print(f"\nTest ROC AUC: {auc:.3f}")

    # Evaluate at two thresholds:
    # - 0.50 standard
    # - 0.35 more FN-sensitive (often preferred in food safety contexts)
    res_050 = eval_at_threshold(y_test_e, y_proba, threshold=0.50, name="LightGBM enriched")
    res_035 = eval_at_threshold(y_test_e, y_proba, threshold=0.35, name="LightGBM enriched")

    metrics_table = pd.concat([res_050, res_035], ignore_index=True)
    print("\nMetrics table:")
    print(metrics_table.to_string(index=False))

    # Detailed reports for each threshold (confusion matrix + classification report)
    for t in [0.50, 0.35]:
        y_pred_t = (y_proba >= t).astype(int)
        cm = confusion_matrix(y_test_e, y_pred_t)
        tn, fp, fn, tp = cm.ravel()

        print("\n" + "=" * 60)
        print(f"Threshold = {t}")
        print(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")
        print(classification_report(y_test_e, y_pred_t, digits=3))
        print("Confusion matrix:\n", cm)

    # =============================================================
    # 10) SHAP explainability (global summary plot)
    # =============================================================
    # TreeExplainer works well for LightGBM
    explainer = shap.TreeExplainer(lgbm)

    # SHAP values for test set
    shap_values = explainer.shap_values(X_test_e)

    # For binary classification:
    # - Some versions return list [class0, class1]
    # - Some return a single array
    shap_vals = shap_values[1] if isinstance(shap_values, list) else shap_values

    # Summary plot shows:
    # - Which features matter most globally
    # - Directionality (high feature values pushing predictions up or down)
    shap.summary_plot(shap_vals, X_test_e)

    # =============================================================
    # 11) Risk prediction for ALL points (used for mapping)
    # =============================================================
    # "risk" = model predicted probability of Listeria presence
    df_enriched["risk"] = lgbm.predict_proba(X_e)[:, 1]

    # =============================================================
    # 12) Create US map: states + hydrology + risk points + high-risk ring
    # =============================================================
    plot_risk_map(
        df_enriched=df_enriched,
        threshold=CFG.high_risk_threshold,
        title="Predicted Listeria Risk with Hydrology Context",
    )


def plot_risk_map(df_enriched: pd.DataFrame, threshold: float, title: str) -> None:
    """
    Plot a US map showing predicted risk points, with hydrology (lakes/rivers)
    and highlighting high-risk points with a red ring.

    Parameters
    ----------
    df_enriched : pd.DataFrame
        Must contain Latitude, Longitude, and risk columns.
    threshold : float
        High-risk threshold (e.g., 0.60)
    title : str
        Figure title
    """
    # ----------------------------
    # Safety checks
    # ----------------------------
    req_cols = ["Latitude", "Longitude", "risk"]
    missing = [c for c in req_cols if c not in df_enriched.columns]
    if missing:
        raise ValueError(f"df_enriched is missing required columns: {missing}")

    df_plot = df_enriched.copy()
    df_plot = df_plot.dropna(subset=["Latitude", "Longitude", "risk"]).reset_index(drop=True)

    # Create a GeoDataFrame of points in WGS84 lat/lon
    gdf_pts = gpd.GeoDataFrame(
        df_plot,
        geometry=gpd.points_from_xy(df_plot["Longitude"], df_plot["Latitude"]),
        crs="EPSG:4326",
    )

    # ----------------------------
    # Load basemap layers (states + hydrology)
    # ----------------------------
    # US states boundaries (GeoJSON)
    us_states = gpd.read_file(
        "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
    ).to_crs("EPSG:4326")

    # Natural Earth hydrology layers:
    lakes = gpd.read_file(
        "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_lakes.geojson"
    ).to_crs("EPSG:4326")

    rivers = gpd.read_file(
        "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_rivers_lake_centerlines.geojson"
    ).to_crs("EPSG:4326")

    # Optional: clip hydrology to US bounds for faster plotting
    xmin, ymin, xmax, ymax = us_states.total_bounds
    bbox_geom = box(xmin, ymin, xmax, ymax)
    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_geom], crs="EPSG:4326")

    try:
        lakes_clip = gpd.overlay(lakes, bbox_gdf, how="intersection")
    except Exception:
        lakes_clip = lakes

    try:
        rivers_clip = gpd.overlay(rivers, bbox_gdf, how="intersection")
    except Exception:
        rivers_clip = rivers

    # ----------------------------
    # Plot
    # ----------------------------
    fig, ax = plt.subplots(figsize=(14, 9))

    # States (light background)
    us_states.plot(ax=ax, color="#f7f7f7", edgecolor="#bdbdbd", linewidth=0.6, zorder=0)

    # Lakes
    lakes_clip.plot(ax=ax, color="#cfe8ff", edgecolor="#9ecae1", linewidth=0.4, alpha=0.55, zorder=1)

    # Rivers
    rivers_clip.plot(ax=ax, color="#8ecae6", linewidth=0.6, alpha=0.7, zorder=2)

    # Risk points (colored by risk)
    sc = ax.scatter(
        df_plot["Longitude"],
        df_plot["Latitude"],
        c=df_plot["risk"],
        cmap="viridis",
        s=38,
        alpha=0.9,
        edgecolor="black",
        linewidth=0.25,
        zorder=3,
    )

    # High-risk ring overlay
    high = df_plot[df_plot["risk"] >= threshold]
    ax.scatter(
        high["Longitude"],
        high["Latitude"],
        s=110,
        facecolors="none",
        edgecolors="red",
        linewidth=1.6,
        zorder=4,
        label=f"High Risk (≥ {threshold:.2f})",
    )

    # Colorbar
    cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Predicted Listeria Risk", rotation=90)

    # Titles and axes
    ax.set_title(title, fontsize=16, pad=10)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Focus on CONUS (adjust if your data include AK/HI)
    ax.set_xlim(-125, -66)
    ax.set_ylim(24, 50)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    ax.legend(loc="lower left")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
