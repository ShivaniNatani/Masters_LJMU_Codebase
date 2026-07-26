"""
Regression tests for the patient-level split - the single most safety-critical
piece of this pipeline, since a bug here means silent train/test leakage in
downstream model results. Nothing else in the repo currently exercises this
function automatically, so these tests are the only guard against a future
edit accidentally reintroducing overlap.
"""
import pandas as pd
import pytest

from preprocessing.patient_splitter import patient_level_split


def _make_df(n_patients: int = 50, studies_per_patient: int = 3) -> pd.DataFrame:
    rows = []
    for p in range(n_patients):
        for s in range(studies_per_patient):
            rows.append({"patient_id": f"P{p}", "study_id": f"P{p}_S{s}", "full_report": "findings text"})
    return pd.DataFrame(rows)


def test_zero_patient_overlap_across_splits():
    df = _make_df()
    train_df, val_df, test_df = patient_level_split(df, seed=42)

    train_p = set(train_df["patient_id"])
    val_p = set(val_df["patient_id"])
    test_p = set(test_df["patient_id"])

    assert not (train_p & val_p)
    assert not (train_p & test_p)
    assert not (val_p & test_p)


def test_every_patient_and_row_is_assigned_exactly_once():
    df = _make_df()
    train_df, val_df, test_df = patient_level_split(df, seed=42)

    assert len(train_df) + len(val_df) + len(test_df) == len(df)

    all_patients = set(df["patient_id"])
    split_patients = set(train_df["patient_id"]) | set(val_df["patient_id"]) | set(test_df["patient_id"])
    assert all_patients == split_patients


def test_split_is_deterministic_given_a_fixed_seed():
    df = _make_df()
    train1, val1, test1 = patient_level_split(df, seed=42)
    train2, val2, test2 = patient_level_split(df, seed=42)

    assert sorted(train1["patient_id"]) == sorted(train2["patient_id"])
    assert sorted(val1["patient_id"]) == sorted(val2["patient_id"])
    assert sorted(test1["patient_id"]) == sorted(test2["patient_id"])


def test_split_ratios_are_approximately_respected():
    df = _make_df(n_patients=200)
    train_df, val_df, test_df = patient_level_split(df, train_ratio=0.70, val_ratio=0.10, test_ratio=0.20, seed=42)

    n_patients = df["patient_id"].nunique()
    train_patients = train_df["patient_id"].nunique()
    val_patients = val_df["patient_id"].nunique()
    test_patients = test_df["patient_id"].nunique()

    assert train_patients == pytest.approx(0.70 * n_patients, abs=1)
    assert val_patients == pytest.approx(0.10 * n_patients, abs=1)
    assert test_patients == pytest.approx(0.20 * n_patients, abs=1)


def test_invalid_ratios_raise_assertion_error():
    df = _make_df(n_patients=10)
    with pytest.raises(AssertionError):
        patient_level_split(df, train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)
