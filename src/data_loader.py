"""
src/data_loader.py
Loads and cleans all Kickstarter CSV shards into a single deduplicated dataframe.
"""
import glob
import json
import os
import re
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def _safe_json_get(val, key, default=None):
    """Extract a key from a JSON-encoded cell value with regex fallback."""
    try:
        if pd.isna(val):
            return default
        d = json.loads(str(val).replace("'", '"'))
        return d.get(key, default)
    except Exception:
        pattern = rf'"{re.escape(key)}"\s*:\s*"([^"]*)"'
        m = re.search(pattern, str(val))
        return m.group(1) if m else default


def load_kickstarter(folder_path: str, verbose: bool = True) -> pd.DataFrame:
    """
    Load all Kickstarter CSV shards from folder_path, deduplicate, filter to
    binary outcome (successful / failed), and return a clean dataframe.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing Kickstarter*.csv files.
    verbose : bool
        Whether to print progress information.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with 'success' binary target column added.
    """
    # 1. Discover files
    pattern = os.path.join(folder_path, "Kickstarter*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No Kickstarter CSV files found at: {pattern}")
    if verbose:
        print(f"Found {len(files)} CSV files")

    # 2. Load and concatenate
    parts = []
    for f in files:
        try:
            parts.append(pd.read_csv(f, low_memory=False))
        except Exception as e:
            if verbose:
                print(f"  [WARN] Could not read {os.path.basename(f)}: {e}")
    df = pd.concat(parts, ignore_index=True)
    if verbose:
        print(f"Raw shape after concat: {df.shape}")

    # 3. Deduplicate on id
    n_before = len(df)
    if "id" in df.columns:
        df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)
    n_removed = n_before - len(df)
    if verbose:
        print(f"Removed {n_removed:,} duplicate rows → {len(df):,} rows")

    # 4. Filter to binary outcome
    all_states = df["state"].value_counts(dropna=False)
    if verbose:
        print("\nAll state counts before filtering:")
        print(all_states.to_string())
    df = df[df["state"].isin(["successful", "failed"])].reset_index(drop=True)
    if verbose:
        print(f"\nAfter binary filter: {len(df):,} rows")

    # 5. Add binary target
    df["success"] = (df["state"] == "successful").astype(int)
    if verbose:
        vc = df["success"].value_counts()
        print(f"\nClass distribution:\n  success=1: {vc.get(1, 0):,} ({vc.get(1, 0)/len(df)*100:.1f}%)")
        print(f"  success=0: {vc.get(0, 0):,} ({vc.get(0, 0)/len(df)*100:.1f}%)")

    # 6. Parse JSON columns
    if "category" in df.columns:
        df["cat_name"] = df["category"].apply(lambda x: _safe_json_get(x, "name"))
        df["cat_parent_name"] = df["category"].apply(lambda x: _safe_json_get(x, "parent_name"))
    if "location" in df.columns:
        df["loc_country"] = df["location"].apply(lambda x: _safe_json_get(x, "country"))
        df["loc_state"] = df["location"].apply(lambda x: _safe_json_get(x, "state"))

    return df


def temporal_split(df: pd.DataFrame, train_frac: float = 0.8):
    """
    Perform a temporal (chronological) train/test split.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with 'launched_at' Unix timestamp column.
    train_frac : float
        Fraction of data (oldest) to use for training.

    Returns
    -------
    train_df, test_df : pd.DataFrame, pd.DataFrame
    """
    df = df.copy()
    df["launched_at"] = pd.to_numeric(df["launched_at"], errors="coerce")
    df = df.sort_values("launched_at").reset_index(drop=True)
    cutoff_idx = int(len(df) * train_frac)
    train_df = df.iloc[:cutoff_idx].copy()
    test_df = df.iloc[cutoff_idx:].copy()
    cutoff_ts = df["launched_at"].iloc[cutoff_idx]
    cutoff_dt = pd.to_datetime(cutoff_ts, unit="s", utc=True)
    print(f"Temporal split cutoff: {cutoff_dt.strftime('%Y-%m-%d')}")
    print(f"Train: {len(train_df):,} rows  |  Test: {len(test_df):,} rows")
    return train_df, test_df
