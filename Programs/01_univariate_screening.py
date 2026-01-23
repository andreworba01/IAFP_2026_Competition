# ============================================================
# IAFP/Cornell Competition — Preprocessing + Univariate Screening
# Dataset: ListeriaSoil_clean.csv
# Author: RiskForgers
# Purpose:
#   1) Load data and print quick diagnostics
#   2) Create binary presence/absence label from isolate counts
#   3) Visualize correlations among numeric predictors
#   4) Run Mann–Whitney U tests + Cliff's delta effect sizes
#   5) Adjust p-values for multiple testing (FDR-BH)
# Output:
#   - Correlation heatmap (on numeric vars)
#   - Ranked table of variables by |Cliff's delta|
# ============================================================
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

from src.config import PATHS


# ---------------------------------------------------------------------
# 1) Reproducibility controls
# ---------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)


# ---------------------------------------------------------------------
# 2) Column names 
# ---------------------------------------------------------------------
# COUNT_COL is the raw isolate count column used to define presence/absence.
# LABEL_COL is created and used as the binary target for screening.
COUNT_COL = "Number of Listeria isolates obtained"
LABEL_COL = "label"


# ---------------------------------------------------------------------
# 3) CLI interface (interactive run options)
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    We use argparse (standard library) so the script runs anywhere
    without extra dependencies.

    Returns
    -------
    argparse.Namespace
        Contains user-selected options.
    """
    parser = argparse.ArgumentParser(
        description="IAFP/Cornell Competition: Univariate screening for Listeria presence"
    )

    # The CSV must be placed under data/raw/ inside the repository.
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="CSV file name located in data/raw/ (e.g., ListeriaSoil_clean.csv)"
    )

    # Show plots interactively (useful in notebooks or local runs).
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display correlation heatmap"
    )

    # Save stats results + metadata for reproducibility.
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save results table + metadata to data/processed/"
    )

    # Save plot and correlation matrix to outputs/ (useful for reports).
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save correlation heatmap PNG and correlation matrix CSV to outputs/"
    )

    # Optionally override seed (still reproducible if documented in metadata).
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for reproducibility (default: 42)"
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# 4) Data loading (repo-relative, portable)
# ---------------------------------------------------------------------
def load_data(filename: str) -> pd.DataFrame:
    """
    Load the dataset from the repository's data/raw/ folder.

    Parameters
    ----------
    filename : str
        Name of the CSV file in data/raw/.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """
    path = PATHS.RAW / filename

    # Fail fast with a clear message (reviewers like explicit errors).
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}\n"
            "Place your dataset under: data/raw/"
        )

    df = pd.read_csv(path)
    return df


def validate_required_columns(df: pd.DataFrame, required: list[str]) -> None:
    """
    Validate that required columns exist in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to validate.
    required : list[str]
        Columns that must exist.

    Raises
    ------
    KeyError
        If any required columns are missing.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


# ---------------------------------------------------------------------
# 5) Label engineering (explicit assumptions)
# ---------------------------------------------------------------------
def add_binary_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary label for Listeria presence.

    Assumption:
      - Presence is defined as COUNT_COL > 0
      - Absence is COUNT_COL == 0
      - Non-numeric values in COUNT_COL are coerced to NaN, then treated as
        NOT present (since NaN > 0 is False). If you want a different policy,
        document it in docs/decisions_log.md.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.

    Returns
    -------
    pd.DataFrame
        Copy of dataset with LABEL_COL added.
    """
    df = df.copy()

    # Convert isolate count to numeric safely.
    df[COUNT_COL] = pd.to_numeric(df[COUNT_COL], errors="coerce")

    # Binary label: 1 if count > 0, else 0
    df[LABEL_COL] = (df[COUNT_COL] > 0).astype(int)

    return df
  # ---------------------------------------------------------------------
# 6) Correlation heatmap (numeric predictors only, no leakage)
# ---------------------------------------------------------------------
def compute_numeric_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Pearson correlation matrix for numeric predictors.

    Important:
      - We exclude the label and isolate count from the correlation plot,
        because including them can artificially inflate correlations and
        mislead interpretation (circularity).

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Correlation matrix of numeric predictors.
    """
    numeric_df = df.select_dtypes(include=[np.number]).copy()

    # Drop target-related columns to avoid circular interpretation.
    numeric_df = numeric_df.drop(columns=[LABEL_COL, COUNT_COL], errors="ignore")

    # If not enough numeric columns, return empty dataframe.
    if numeric_df.shape[1] < 2:
        return pd.DataFrame()

    return numeric_df.corr(method="pearson")


def plot_correlation_heatmap(corr: pd.DataFrame, show: bool, save: bool) -> None:
    """
    Plot and optionally save correlation heatmap.

    Parameters
    ----------
    corr : pd.DataFrame
        Correlation matrix.
    show : bool
        If True, display the plot.
    save : bool
        If True, save plot and correlation matrix under outputs/.
    """
    if corr.empty:
        print("Not enough numeric predictors to compute correlation heatmap.")
        return

    # Create outputs folder if saving.
    if save:
        PATHS.OUTPUTS.mkdir(parents=True, exist_ok=True)
        corr.to_csv(PATHS.OUTPUTS / "correlation_matrix.csv", index=True)

    plt.figure(figsize=(14, 12))
    sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.5)
    plt.title("Correlation Matrix of Numeric Predictors (excluding target columns)")
    plt.tight_layout()

    if save:
        plt.savefig(PATHS.OUTPUTS / "correlation_heatmap.png", dpi=300)

    if show:
        plt.show()
    else:
        # Close figure to avoid memory leaks in batch runs/CI.
        plt.close()


# ---------------------------------------------------------------------
# 7) Effect size: Cliff's delta (interpretable for nonparametric comparisons)
# ---------------------------------------------------------------------
def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Cliff's delta effect size between two groups.

    Definition:
      delta = P(X > Y) - P(X < Y)

    Interpretation:
      - delta ~ 0: overlap (little difference)
      - delta > 0: values tend to be higher in X (e.g., present group)
      - delta < 0: values tend to be lower in X

    Notes:
      - This implementation is O(n*m). Fine for small/medium datasets.
      - If dataset is large (e.g., 10k+ rows), we can replace with a faster
        rank-based implementation.

    Parameters
    ----------
    x, y : np.ndarray
        Arrays of values for group X and group Y.

    Returns
    -------
    float
        Cliff's delta in [-1, 1] or NaN if empty groups.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    nx, ny = x.size, y.size
    if nx == 0 or ny == 0:
        return np.nan

    gt = sum(ix > iy for ix in x for iy in y)
    lt = sum(ix < iy for ix in x for iy in y)
    return (gt - lt) / (nx * ny)


# ---------------------------------------------------------------------
# 8) Univariate screening: Mann–Whitney U + Cliff's delta + FDR correction
# ---------------------------------------------------------------------
def univariate_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare numeric predictors between absence vs presence groups.

    Test:
      - Mann–Whitney U (two-sided) for distribution shift
    Effect:
      - Cliff's delta (direction + magnitude)

    Multiple testing:
      - FDR-BH (Benjamini–Hochberg) applied to p-values

    Returns
    -------
    pd.DataFrame
        Ranked results sorted by absolute Cliff's delta descending.
    """
    # Split groups based on label
    df_absent = df[df[LABEL_COL] == 0]
    df_present = df[df[LABEL_COL] == 1]

    # Select numeric variables only (prevents accidental inclusion of strings/IDs)
    numeric_vars = df.select_dtypes(include=[np.number]).columns

    # Drop outcome-related columns to avoid leakage/circularity
    numeric_vars = numeric_vars.drop([LABEL_COL, COUNT_COL], errors="ignore")

    results = []

    for var in numeric_vars:
        # Drop missing values within each group (common, explicit policy)
        x0 = df_absent[var].dropna()
        x1 = df_present[var].dropna()

        # If one group is too small, skip or record; here we record minimal info.
        if x0.size < 3 or x1.size < 3:
            results.append({
                "Variable": var,
                "n_absent": x0.size,
                "n_present": x1.size,
                "Median_Absent": np.nan if x0.size == 0 else np.median(x0),
                "Median_Present": np.nan if x1.size == 0 else np.median(x1),
                "Median_Difference": np.nan,
                "Cliffs_Delta": np.nan,
                "p_value": np.nan,
                "note": "Skipped: too few observations in one group"
            })
            continue

        # Mann–Whitney U (nonparametric)
        stat, p = mannwhitneyu(x0, x1, alternative="two-sided")

        # Effect size: positive means higher in presence group
        delta = cliffs_delta(x1.values, x0.values)

        results.append({
            "Variable": var,
            "n_absent": x0.size,
            "n_present": x1.size,
            "Median_Absent": np.median(x0),
            "Median_Present": np.median(x1),
            "Median_Difference": np.median(x1) - np.median(x0),
            "Cliffs_Delta": delta,
            "p_value": p,
            "note": ""
        })

    stats_df = pd.DataFrame(results)

    # Apply FDR correction to valid p-values only (ignore NaNs)
    stats_df["p_adj_fdr"] = np.nan
    mask = stats_df["p_value"].notna()

    if mask.sum() > 0:
        stats_df.loc[mask, "p_adj_fdr"] = multipletests(
            stats_df.loc[mask, "p_value"].values,
            method="fdr_bh"
        )[1]

    # Rank by absolute effect size
    stats_df = stats_df.sort_values(
        by="Cliffs_Delta",
        key=lambda s: s.abs(),
        ascending=False
    ).reset_index(drop=True)

    return stats_df


# ---------------------------------------------------------------------
# 9) Saving outputs + metadata (reproducibility requirement)
# ---------------------------------------------------------------------
def save_results(stats_df: pd.DataFrame, meta: dict) -> None:
    """
    Save results + metadata into data/processed/.

    We store metadata (JSON) so anyone can reproduce:
      - seed
      - file name
      - timestamp
      - label balance
      - number of predictors evaluated
      - dataset shape
    """
    PATHS.PROCESSED.mkdir(parents=True, exist_ok=True)

    results_path = PATHS.PROCESSED / "univariate_screening_results.csv"
    meta_path = PATHS.PROCESSED / "univariate_screening_metadata.json"

    stats_df.to_csv(results_path, index=False)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved results: {results_path}")
    print(f"Saved metadata: {meta_path}")


# ---------------------------------------------------------------------
# 10) Main execution (single entry point)
# ---------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    # Allow user to override seed (still reproducible since we log it)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Load + validate
    df = load_data(args.input)
    validate_required_columns(df, [COUNT_COL])

    # Basic diagnostics (useful for reviewers and debugging)
    print("\nDataset loaded:")
    print(df.head())
    print("\nDataset info:")
    print(df.info())
    print("\nShape:", df.shape)

    # Create label
    df = add_binary_label(df)

    # Label balance summary
    label_dist = df[LABEL_COL].value_counts(dropna=False)
    label_prop = df[LABEL_COL].value_counts(normalize=True, dropna=False)

    print("\nLabel counts:")
    print(label_dist)
    print("\nLabel proportions:")
    print(label_prop)

    # Correlation
    corr = compute_numeric_correlation(df)
    plot_correlation_heatmap(corr, show=args.show_plots, save=args.save_plots)

    # Univariate analysis
    stats_df = univariate_analysis(df)

    print("\nTop 15 predictors by |Cliff's delta|:")
    print(stats_df.head(15))

    # Save outputs if requested
    if args.save_results:
        meta = {
            "timestamp_utc": datetime.utcnow().isoformat(),
            "input_file": args.input,
            "seed": args.seed,
            "dataset_shape": list(df.shape),
            "label_counts": label_dist.to_dict(),
            "label_proportions": {k: float(v) for k, v in label_prop.to_dict().items()},
            "n_predictors_tested": int(stats_df["Variable"].nunique()),
            "notes": [
                f"Label defined as ({COUNT_COL} > 0).",
                "Correlation computed on numeric predictors excluding target columns.",
                "Univariate test: Mann–Whitney U (two-sided); Effect size: Cliff's delta.",
                "Multiple-testing correction: FDR (Benjamini–Hochberg)."
            ]
        }
        save_results(stats_df, meta)


if __name__ == "__main__":
    main()
