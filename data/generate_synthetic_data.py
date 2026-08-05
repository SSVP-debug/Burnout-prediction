"""
Synthetic employee burnout data generator.

DATA PROVENANCE (read this before trusting any number downstream):
--------------------------------------------------------------------
This dataset is 100% synthetic. It is NOT real employee data, was NOT
collected from any company, and does NOT represent any real person.
It exists to give the modeling pipeline something realistic to train
and evaluate on, in place of real HR data (which is either private,
protected by employment law, or simply unavailable publicly).

WHY SYNTHETIC:
There is no legitimate public dataset combining per-employee behavioral
signals (meeting load, sleep, after-hours work, mood) with a burnout
label. Any repo claiming to use "real" data for this without a named,
verifiable source should be treated with suspicion. This one says
plainly what it is.

GENERATION METHOD:
1. 18 raw features are sampled independently from distributions chosen
   to be individually plausible (e.g. sleep hours ~ Normal(6.8, 1.1),
   meetings/day ~ Poisson(4)).
2. A continuous "strain score" is computed as a WEIGHTED, DOCUMENTED
   linear combination of those features (weights below) plus Gaussian
   noise, so the label is not a trivial function of any single feature
   and the classification task is genuinely learnable but not perfectly
   separable — the point is to mimic real-world signal + noise, not to
   hand the model an easy 100% accuracy.
3. The strain score is binned into 3 classes (Low / Moderate / High)
   using fixed quantile-derived thresholds, producing a roughly
   balanced 3-class problem.

This is NOT a simulation of any published psychological burnout model
(e.g. Maslach Burnout Inventory) — it's a synthetic proxy built for
this project. That distinction matters and is stated here explicitly.

Reproducibility: fixed random seed (42). Re-running this script
produces byte-identical output.
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_SAMPLES = 4000

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

# Weights applied to (z-scored) features to build the latent strain score.
# Positive weight = higher feature value increases burnout risk.
# Negative weight = higher feature value is protective.
# These are hand-set, documented assumptions -- not fit from real data.
STRAIN_WEIGHTS = {
    "meetings_count": 0.55,
    "total_meeting_minutes": 0.60,
    "back_to_back_meetings": 0.70,
    "avg_meeting_gap_minutes": -0.35,
    "deep_work_minutes": -0.60,
    "context_switches": 0.50,
    "emails_after_hours": 0.45,
    "after_hours_work_minutes": 0.65,
    "sleep_hours_prev_night": -0.75,
    "exercise_minutes": -0.40,
    "mood_prev_day": -0.55,
    "tasks_completed": -0.20,
    "overdue_tasks": 0.50,
    "interruptions_count": 0.40,
    "caffeine_cups": 0.25,
    "wfh": -0.10,
    "commute_minutes": 0.25,
    "sick_or_caregiving": 0.30,
}


def _sample_raw_features(rng: np.random.Generator, n: int) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "meetings_count": rng.poisson(4, n),
            "total_meeting_minutes": rng.normal(180, 70, n).clip(0, 600),
            "back_to_back_meetings": rng.poisson(1.5, n).clip(0, 10),
            "avg_meeting_gap_minutes": rng.normal(25, 15, n).clip(0, 120),
            "deep_work_minutes": rng.normal(150, 80, n).clip(0, 480),
            "context_switches": rng.poisson(8, n),
            "emails_after_hours": rng.poisson(3, n),
            "after_hours_work_minutes": rng.exponential(35, n).clip(0, 300),
            "sleep_hours_prev_night": rng.normal(6.8, 1.1, n).clip(2, 10),
            "exercise_minutes": rng.exponential(20, n).clip(0, 180),
            "mood_prev_day": rng.integers(1, 6, n),  # 1 (bad) - 5 (great)
            "tasks_completed": rng.poisson(6, n),
            "overdue_tasks": rng.poisson(1.2, n),
            "interruptions_count": rng.poisson(5, n),
            "caffeine_cups": rng.poisson(2, n),
            "wfh": rng.integers(0, 2, n),  # 0/1
            "commute_minutes": rng.exponential(25, n).clip(0, 120),
            "sick_or_caregiving": rng.binomial(1, 0.08, n),
        }
    )
    return df[FEATURE_COLUMNS]


def _compute_labels(df: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    z = (df - df.mean()) / df.std()
    strain = sum(z[col] * w for col, w in STRAIN_WEIGHTS.items())
    strain += rng.normal(0, 1.0, len(df))  # irreducible noise

    q1, q2 = strain.quantile([0.40, 0.75])
    labels = pd.cut(
        strain,
        bins=[-np.inf, q1, q2, np.inf],
        labels=["Low", "Moderate", "High"],
    )
    return labels.astype(str)


def generate(n_samples: int = N_SAMPLES, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = _sample_raw_features(rng, n_samples)
    df["burnout_risk"] = _compute_labels(df, rng)
    df.insert(0, "employee_id", [f"E{i:05d}" for i in range(1, n_samples + 1)])
    return df


if __name__ == "__main__":
    data = generate()
    out_path = "data/burnout_data.csv"
    data.to_csv(out_path, index=False)
    print(f"Wrote {len(data)} rows to {out_path}")
    print(data["burnout_risk"].value_counts(normalize=True).round(3))
