"""Load the burnout dataset and produce a stratified train/test split."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "burnout_data.csv"

FEATURE_COLUMNS = [
    "meetings_count",
    "total_meeting_minutes",
    "back_to_back_meetings",
    "avg_meeting_gap_minutes",
    "deep_work_minutes",
    "context_switches",
    "emails_after_hours",
    "after_hours_work_minutes",
    "sleep_hours_prev_night",
    "exercise_minutes",
    "mood_prev_day",
    "tasks_completed",
    "overdue_tasks",
    "interruptions_count",
    "caffeine_cups",
    "wfh",
    "commute_minutes",
    "sick_or_caregiving",
]

TARGET_COLUMN = "burnout_risk"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"No dataset found at {path}. Run "
            "`python data/generate_synthetic_data.py` first."
        )
    return pd.read_csv(path)


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Stratified train/test split on the target label."""
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def load_train_test(path: Path = DATA_PATH, test_size: float = 0.2, random_state: int = 42):
    df = load_data(path)
    return split_data(df, test_size=test_size, random_state=random_state)
