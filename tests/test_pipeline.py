"""Automated tests: they prove the cleaning, audit and forecasting logic behaves as claimed.

Run with:  pytest
"""
import numpy as np
import pandas as pd
import pytest

from src import config as C
from src import data_quality as dq
from src import forecast as fc
from src.cleaning import clean_opportunities, normalise_industry, normalise_stage
from src.generate_data import main as generate


@pytest.fixture(scope="session")
def raw():
    if not (C.DATA_RAW / "opportunities.csv").exists():
        generate()
    return (pd.read_csv(C.DATA_RAW / "opportunities.csv"), pd.read_csv(C.DATA_RAW / "accounts.csv"),
            pd.read_csv(C.DATA_RAW / "targets.csv"))


@pytest.fixture(scope="session")
def clean(raw):
    return clean_opportunities(raw[0])


# ---------------------------------------------------------------- cleaning
@pytest.mark.parametrize("label,expected", [("closed won ", "Won"), ("LOST", "Lost"), ("Proposal/Quote", "Proposal"),
                                            ("negotiation", "Negotiation"), ("Qualified", "Qualification"),
                                            ("Prospect", "Prospecting")])
def test_stage_normalisation(label, expected):
    assert normalise_stage(label) == expected


def test_unknown_stage_is_nan():
    assert pd.isna(normalise_stage("???")) and pd.isna(normalise_stage(None))


def test_industry_normalisation():
    assert normalise_industry(" Oil and Gas ") == "Oil & Gas"
    assert normalise_industry("MINING ") == "Mining"
    assert normalise_industry("Space Mining") == "Unknown"


def test_clean_output_is_consistent(clean):
    df, log = clean
    assert df["opportunity_id"].is_unique
    assert set(df["stage"].dropna()) <= set(C.CANONICAL_STAGES)
    assert df["stage"].notna().all()
    assert df["owner_id"].notna().all()
    assert (df["amount"].dropna() > 0).all()
    closed = df[df["is_won"] | df["is_lost"]]
    assert closed["close_date"].notna().all()
    assert (closed["close_date"] >= closed["created_date"]).all()
    assert log["duplicate_rows_removed"] > 0


def test_stale_and_overdue_deals_never_in_active_pipeline(clean):
    df, _ = clean
    assert not (df["in_active_pipeline"] & (df["dq_stale"] | df["dq_overdue"])).any()
    assert not (df["in_active_pipeline"] & ~df["is_open"]).any()


# ---------------------------------------------------------------- audit
def test_sql_and_pandas_audits_agree(raw):
    p = dq.run_pandas_audit(raw[0], raw[1])
    s = dq.run_sql_audit(C.DATA_RAW / "opportunities.csv", C.DATA_RAW / "accounts.csv")
    assert dq.reconcile(p, s) == []
    assert len(p) == len(s) == 12


def test_audit_finds_the_injected_problems(raw):
    p = dq.run_pandas_audit(raw[0], raw[1]).set_index("check_id")
    assert (p["rows_affected"] > 0).all()


# ---------------------------------------------------------------- forecasting
@pytest.fixture(scope="module")
def series():
    idx = pd.period_range("2024-01", periods=33, freq="M")
    rng = np.random.default_rng(0)
    return pd.Series(1_000_000 + 20_000 * np.arange(33) + rng.normal(0, 50_000, 33), index=idx)


def test_backtest_does_not_leak_future_data(series):
    """Changing the final observation must not change ANY backtest forecast."""
    preds_a, _ = fc.backtest(series, 6)
    altered = series.copy()
    altered.iloc[-1] *= 10
    preds_b, _ = fc.backtest(altered, 6)
    assert np.allclose(preds_a["forecast"], preds_b["forecast"])


def test_regression_recovers_a_clean_trend():
    idx = pd.period_range("2024-01", periods=30, freq="M")
    y = pd.Series(1_000_000 + 10_000 * np.arange(30), index=idx, dtype=float)
    pred = fc.predict_trend_seasonal(y, 3)
    assert np.allclose(pred, [1_000_000 + 10_000 * t for t in (30, 31, 32)], rtol=1e-6)


def test_seasonal_naive_returns_same_month_last_year(series):
    pred = fc.predict_seasonal_naive(series, 2)
    assert pred[0] == series.iloc[-12] and pred[1] == series.iloc[-11]


def test_full_forecast_shapes(clean, raw):
    df, _ = clean
    res = fc.run_forecast(df, raw[2])
    f = res["forecast"]
    assert len(f) == C.FORECAST_MONTHS
    assert (f["lower_80"] <= f["statistical_forecast"]).all() and (f["statistical_forecast"] <= f["upper_80"]).all()
    assert f["period"].iloc[0] == C.SNAPSHOT_DATE.to_period("M")
