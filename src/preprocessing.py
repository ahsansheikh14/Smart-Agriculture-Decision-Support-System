"""
Data loading and preparation for Smart Agriculture DSS.
Phase 2: load, inspect, split, and scale data for 3 models.
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Paths: this file is in src/, so project root is one level up
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "crop_recommendation.csv"

# Columns we will use as input features in all 3 models
FEATURE_COLUMNS = [
    "N", "P", "K", "temperature", "humidity", "ph", "rainfall"
]

# Column that stores crop name (Decision Tree target)
TARGET_CROP = "label"

# Column for Linear Regression (we create this in Step 2)
TARGET_YIELD = "yield"

# 80% train, 20% test — same split every time because of random_state
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_data():
    """Read the CSV file and return a pandas DataFrame."""
    df = pd.read_csv(DATA_PATH)
    return df


def inspect_data(df):
    """Print basic information so we can trust the data before training."""
    print("=== Dataset Overview ===")
    print(f"Shape (rows, columns): {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nNumber of different crops: {df[TARGET_CROP].nunique()}")
    print(f"\nTop 5 most common crops:")
    print(df[TARGET_CROP].value_counts().head())


def add_yield_column(df):
    """
    Create a yield column for regression.

    The original Kaggle CSV has no yield. We build a simple formula from
    soil nutrients + rainfall so Linear Regression has a numeric target.
    (Explain this clearly in viva.)
    """
    df = df.copy()
    df[TARGET_YIELD] = (
        2.5
        + 0.05 * df["N"]
        + 0.04 * df["P"]
        + 0.03 * df["K"]
        + 0.015 * df["rainfall"]
        - 0.08 * (df["ph"] - 6.5).abs()
    )
    return df


def prepare_data():
    """
    Prepare everything our 3 models need:
    - X = features (7 columns)
    - y_crop = crop name (classification)
    - y_yield = yield value (regression)
    - train/test splits
    - scaled X for clustering
    """
    df = load_data()
    df = add_yield_column(df)

    # X = inputs, y = outputs
    X = df[FEATURE_COLUMNS]
    y_crop = df[TARGET_CROP]
    y_yield = df[TARGET_YIELD]

    # Decision Tree: train on 80%, test on 20%
    X_train, X_test, y_crop_train, y_crop_test = train_test_split(
        X,
        y_crop,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_crop,  # keep crop balance in train and test
    )

    # Linear Regression: separate split (same settings)
    X_train_r, X_test_r, y_yield_train, y_yield_test = train_test_split(
        X,
        y_yield,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    # KMeans: scale features so N, P, K, temperature, etc. are comparable
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return {
        "df": df,
        "X": X,
        "y_crop": y_crop,
        "y_yield": y_yield,
        "X_scaled": X_scaled,
        "scaler": scaler,
        "X_train": X_train,
        "X_test": X_test,
        "y_crop_train": y_crop_train,
        "y_crop_test": y_crop_test,
        "X_train_r": X_train_r,
        "X_test_r": X_test_r,
        "y_yield_train": y_yield_train,
        "y_yield_test": y_yield_test,
    }


def inspect_splits(data):
    """Print sizes so we can verify Step 2 worked."""
    print("\n=== Step 2: Prepared Data ===")
    print(f"Total rows: {len(data['X'])}")
    print(f"Features (X) columns: {list(data['X'].columns)}")
    print(f"\nClassification split:")
    print(f"  Train: {len(data['X_train'])} rows")
    print(f"  Test:  {len(data['X_test'])} rows")
    print(f"\nRegression split:")
    print(f"  Train: {len(data['X_train_r'])} rows")
    print(f"  Test:  {len(data['X_test_r'])} rows")
    print(f"\nYield sample (first 5):")
    print(data["y_yield"].head().values)
    print(f"\nScaled data shape (for clustering): {data['X_scaled'].shape}")


if __name__ == "__main__":
    df = load_data()
    inspect_data(df)

    data = prepare_data()
    inspect_splits(data)
