"""
src/features.py
All feature engineering functions for the Kickstarter ML project.
Each feature has its own function with a docstring.
Master function build_features(df) calls all of them in order.
"""
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Individual feature functions
# ---------------------------------------------------------------------------

def feat_log_goal(df: pd.DataFrame) -> pd.DataFrame:
    """
    log_goal = log(1 + goal)
    Goal has extreme right skew (max ~$1B). Log transform compresses the
    range and prevents extreme outliers from dominating tree splits and
    linear model coefficients.
    """
    df = df.copy()
    goal = pd.to_numeric(df["goal"], errors="coerce").fillna(0).clip(lower=0)
    df["log_goal"] = np.log1p(goal)
    return df


def feat_duration_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    duration_days = (deadline - launched_at) / 86400
    Campaign length in days. Capped at 90 (Kickstarter's maximum allowed).
    Short campaigns (30 days) historically outperform long ones.
    """
    df = df.copy()
    launched = pd.to_numeric(df["launched_at"], errors="coerce")
    deadline = pd.to_numeric(df["deadline"], errors="coerce")
    df["duration_days"] = ((deadline - launched) / 86400).clip(0, 90).astype(float)
    return df


def feat_prep_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    prep_days = (launched_at - created_at) / 86400
    Days between account creation and campaign launch. Proxy for preparation effort.
    Winsorised at 99th percentile to handle creators who created accounts years
    before launching (these would be outliers unrelated to preparation).
    """
    df = df.copy()
    created = pd.to_numeric(df.get("created_at", pd.Series(np.nan, index=df.index)), errors="coerce")
    launched = pd.to_numeric(df["launched_at"], errors="coerce")
    raw = ((launched - created) / 86400).clip(lower=0)
    cap = raw.quantile(0.99)
    df["prep_days"] = raw.clip(upper=cap).astype(float)
    return df


def feat_has_video(df: pd.DataFrame) -> pd.DataFrame:
    """
    has_video = 1 if video column is not null, else 0.
    Campaigns with a video demo raise 105% more on average (Kickstarter data).
    Missing video is informative (no video), not missing-at-random.
    """
    df = df.copy()
    if "video" in df.columns:
        df["has_video"] = df["video"].notna().astype(int)
    else:
        df["has_video"] = 0
    return df


def feat_blurb_length(df: pd.DataFrame) -> pd.DataFrame:
    """
    blurb_length = number of characters in the campaign short description.
    Longer, more descriptive blurbs may signal more effort and clarity.
    """
    df = df.copy()
    df["blurb_length"] = df["blurb"].fillna("").str.len().astype(int)
    return df


def feat_name_length(df: pd.DataFrame) -> pd.DataFrame:
    """
    name_length = number of characters in the campaign title.
    Very short or very long titles may signal poor presentation.
    """
    df = df.copy()
    df["name_length"] = df["name"].fillna("").str.len().astype(int)
    return df


def feat_blurb_word_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    blurb_word_count = number of words in the blurb.
    Complements blurb_length — captures average word length variation.
    """
    df = df.copy()
    df["blurb_word_count"] = (
        df["blurb"].fillna("").str.split().str.len().fillna(0).astype(int)
    )
    return df


def feat_name_has_number(df: pd.DataFrame) -> pd.DataFrame:
    """
    name_has_number = 1 if campaign name contains a digit.
    Names like "Volume 2" or "Season 3" suggest an established creator
    with prior work and an existing audience.
    """
    df = df.copy()
    df["name_has_number"] = (
        df["name"].fillna("").str.contains(r"\d", regex=True).astype(int)
    )
    return df


def feat_goal_is_round(df: pd.DataFrame) -> pd.DataFrame:
    """
    goal_is_round = 1 if goal is divisible by 1000.
    Round goals ($10,000) may indicate less precise budgeting
    compared to specific goals ($8,750), suggesting lower preparation quality.
    """
    df = df.copy()
    goal = pd.to_numeric(df["goal"], errors="coerce").fillna(0)
    df["goal_is_round"] = ((goal % 1000) == 0).astype(int)
    return df


def feat_is_usd(df: pd.DataFrame) -> pd.DataFrame:
    """
    is_usd = 1 if currency is USD.
    USD campaigns dominate the platform and have the largest potential audience.
    Non-USD campaigns may face currency conversion friction from backers.
    """
    df = df.copy()
    if "currency" in df.columns:
        df["is_usd"] = (df["currency"] == "USD").astype(int)
    else:
        df["is_usd"] = 0
    return df


def feat_launch_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    launched_month (1-12), launched_dayofweek (0=Mon, 6=Sun), launched_year.
    Seasonal effects matter: Q4 pre-holiday launches may perform differently.
    Year captures platform maturity, economic conditions, and trend evolution.
    Tuesday/Wednesday launches have historically higher success rates.
    """
    df = df.copy()
    launched = pd.to_datetime(
        pd.to_numeric(df["launched_at"], errors="coerce"), unit="s", utc=True
    )
    df["launched_month"] = launched.dt.month.astype(float)
    df["launched_dayofweek"] = launched.dt.dayofweek.astype(float)
    df["launched_year"] = launched.dt.year.astype(float)
    return df


def feat_staff_pick_binary(df: pd.DataFrame) -> pd.DataFrame:
    """
    staff_pick as 0/1 integer.
    Staff-picked campaigns are editorially featured by Kickstarter before launch.
    """
    df = df.copy()
    if "staff_pick" in df.columns:
        sp = df["staff_pick"]
        if sp.dtype == bool:
            df["staff_pick"] = sp.astype(int)
        else:
            df["staff_pick"] = sp.map(
                {"True": 1, "False": 0, True: 1, False: 0}
            ).fillna(0).astype(int)
    else:
        df["staff_pick"] = 0
    return df


def feat_goal_per_day(df: pd.DataFrame) -> pd.DataFrame:
    """
    goal_per_day = goal / duration_days
    The daily fundraising target. Captures the intensity of the ask.
    A campaign asking for $100,000 in 5 days is far harder than the same
    goal over 60 days. Division by zero → 0.
    """
    df = df.copy()
    goal = pd.to_numeric(df["goal"], errors="coerce").fillna(0)
    dur = df.get("duration_days", pd.Series(np.nan, index=df.index))
    dur = pd.to_numeric(dur, errors="coerce").replace(0, np.nan)
    df["goal_per_day"] = (goal / dur).fillna(0).astype(float)
    return df


def feat_is_california(df: pd.DataFrame) -> pd.DataFrame:
    """
    is_california = 1 if the campaign's location state is CA.
    California (especially SF/LA) has the largest creator ecosystem on Kickstarter
    and may benefit from stronger network and press effects.
    """
    df = df.copy()
    if "loc_state" in df.columns:
        df["is_california"] = (df["loc_state"] == "CA").astype(int)
    else:
        df["is_california"] = 0
    return df


# ---------------------------------------------------------------------------
# Target encoding for cat_name (must be fit on training data only)
# ---------------------------------------------------------------------------

def fit_target_encoder(train_df: pd.DataFrame, col: str, target: str = "success",
                        smoothing: float = 10.0) -> dict:
    """
    Fit a smoothed target encoder on training data.
    smoothing controls how much we shrink toward the global mean for rare categories.
    Formula: encoded = (n * cat_mean + smoothing * global_mean) / (n + smoothing)
    This MUST be fit on training data only to avoid target leakage.
    """
    global_mean = train_df[target].mean()
    stats = train_df.groupby(col)[target].agg(["mean", "count"])
    encoder = {}
    for cat, row in stats.iterrows():
        n = row["count"]
        cat_mean = row["mean"]
        encoded = (n * cat_mean + smoothing * global_mean) / (n + smoothing)
        encoder[cat] = encoded
    encoder["__global_mean__"] = global_mean
    return encoder


def apply_target_encoder(df: pd.DataFrame, col: str, encoder: dict,
                          new_col: str = None) -> pd.DataFrame:
    """Apply a fitted target encoder to a dataframe."""
    df = df.copy()
    out_col = new_col or f"{col}_te"
    global_mean = encoder.get("__global_mean__", 0.5)
    df[out_col] = df[col].map(encoder).fillna(global_mean)
    return df


# ---------------------------------------------------------------------------
# One-hot encoding helpers (fit on train, apply to both)
# ---------------------------------------------------------------------------

def fit_ohe_country(train_df: pd.DataFrame, min_count: int = 500) -> dict:
    """
    Identify countries with >= min_count campaigns.
    Others are grouped into 'country_other'.
    Returns a dict with 'keep_countries' list.
    """
    vc = train_df["country"].value_counts()
    keep = vc[vc >= min_count].index.tolist()
    return {"keep_countries": keep}


def apply_ohe_country(df: pd.DataFrame, encoder: dict) -> pd.DataFrame:
    """Apply country grouping + one-hot encoding."""
    df = df.copy()
    keep = encoder["keep_countries"]
    df["country_grouped"] = df["country"].where(df["country"].isin(keep), other="Other")
    dummies = pd.get_dummies(df["country_grouped"], prefix="country").astype(int)
    df = pd.concat([df, dummies], axis=1)
    return df


def fit_ohe_cat_parent(train_df: pd.DataFrame) -> list:
    """Return sorted list of unique cat_parent_name values from training set."""
    return sorted(train_df["cat_parent_name"].dropna().unique().tolist())


def apply_ohe_cat_parent(df: pd.DataFrame, categories: list) -> pd.DataFrame:
    """One-hot encode cat_parent_name using training categories."""
    df = df.copy()
    df["cat_parent_name"] = df["cat_parent_name"].fillna("Unknown")
    for cat in categories:
        col = f"parent_{cat.replace(' ', '_').replace('/', '_')}"
        df[col] = (df["cat_parent_name"] == cat).astype(int)
    return df


# ---------------------------------------------------------------------------
# Master function
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame,
                   train_df: pd.DataFrame = None,
                   encoders: dict = None) -> pd.DataFrame:
    """
    Build all features. If train_df is provided, encoders are fit on it.
    If encoders dict is provided, they are used directly (for test set).
    Returns the engineered dataframe and the encoders dict.
    """
    df = feat_log_goal(df)
    df = feat_duration_days(df)
    df = feat_prep_days(df)
    df = feat_has_video(df)
    df = feat_blurb_length(df)
    df = feat_name_length(df)
    df = feat_blurb_word_count(df)
    df = feat_name_has_number(df)
    df = feat_goal_is_round(df)
    df = feat_is_usd(df)
    df = feat_launch_time(df)
    df = feat_staff_pick_binary(df)
    df = feat_goal_per_day(df)
    df = feat_is_california(df)

    if encoders is None and train_df is not None:
        encoders = {}
        # Target encoding for cat_name
        if "cat_name" in df.columns and "success" in df.columns:
            encoders["cat_name_te"] = fit_target_encoder(train_df, "cat_name")
        # Country OHE
        if "country" in df.columns:
            encoders["country_ohe"] = fit_ohe_country(train_df)
        # Parent category OHE
        if "cat_parent_name" in df.columns:
            encoders["cat_parent_cats"] = fit_ohe_cat_parent(train_df)

    if encoders:
        if "cat_name_te" in encoders and "cat_name" in df.columns:
            df = apply_target_encoder(df, "cat_name", encoders["cat_name_te"],
                                       new_col="cat_name_encoded")
        if "country_ohe" in encoders and "country" in df.columns:
            df = apply_ohe_country(df, encoders["country_ohe"])
        if "cat_parent_cats" in encoders and "cat_parent_name" in df.columns:
            df = apply_ohe_cat_parent(df, encoders["cat_parent_cats"])

    return df, encoders
