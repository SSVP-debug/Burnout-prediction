import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.generate_synthetic_data import FEATURE_COLUMNS, generate
from src.data_loader import TARGET_COLUMN, split_data
from src.preprocessing import build_scaler


def test_generate_returns_expected_columns():
    df = generate(n_samples=100, seed=1)
    for col in FEATURE_COLUMNS:
        assert col in df.columns
    assert TARGET_COLUMN in df.columns
    assert "employee_id" in df.columns


def test_generate_is_deterministic():
    df1 = generate(n_samples=50, seed=7)
    df2 = generate(n_samples=50, seed=7)
    pd.testing.assert_frame_equal(df1, df2)


def test_generate_labels_are_valid_classes():
    df = generate(n_samples=200, seed=3)
    assert set(df[TARGET_COLUMN].unique()) <= {"Low", "Moderate", "High"}


def test_no_missing_values():
    df = generate(n_samples=200, seed=3)
    assert not df.isnull().values.any()


def test_split_data_is_stratified_and_disjoint():
    df = generate(n_samples=500, seed=5)
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=5)

    assert len(X_train) + len(X_test) == len(df)
    assert set(X_train.index).isdisjoint(set(X_test.index))

    train_prop = y_train.value_counts(normalize=True).sort_index()
    test_prop = y_test.value_counts(normalize=True).sort_index()
    assert np.allclose(train_prop.values, test_prop.values, atol=0.05)


def test_scaler_produces_zero_mean_unit_variance():
    df = generate(n_samples=300, seed=9)
    scaler = build_scaler()
    scaled = scaler.fit_transform(df[FEATURE_COLUMNS])
    assert np.allclose(scaled.mean(axis=0), 0, atol=1e-8)
    assert np.allclose(scaled.std(axis=0), 1, atol=1e-2)
