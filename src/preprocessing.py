"""Preprocessing shared by every model in this project.

All features here are already numeric (no categorical encoding needed),
so preprocessing is limited to standardization. Standardization is a
no-op for tree models but required for the Logistic Regression baseline
to converge properly and be comparable -- kept in one shared place so
train.py, evaluate.py, and app.py can never drift out of sync with each
other about how inputs are scaled.
"""

from sklearn.preprocessing import StandardScaler


def build_scaler() -> StandardScaler:
    return StandardScaler()
