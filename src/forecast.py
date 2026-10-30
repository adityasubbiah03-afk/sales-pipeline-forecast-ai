"""Revenue forecasting: statistical models, an honest backtest, and a weighted-pipeline view.

Two complementary views of "what will we win next month?"
  1. STATISTICAL  - learn the pattern (trend + seasonality) from past monthly won revenue.
  2. PIPELINE     - bottom-up: sum(open deal amount x stage probability) by expected close month.

The statistical models are backtested with a ROLLING-ORIGIN test: for each of the last N months
we pretend we are standing at the end of the previous month, train only on data available then,
forecast one month ahead and compare with what really happened. No future data ever leaks in.

Limitation (documented in the README): the pipeline view cannot be backtested here because that
needs historical snapshots of the open pipeline, which a single CRM extract does not contain.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src import config as C


# ------------------------------------------------------------------ data prep
def monthly_won_revenue(df: pd.DataFrame, snapshot=C.SNAPSHOT_DATE) -> pd.Series:
    """Won revenue per calendar month, up to the last COMPLETE month before the snapshot."""
    won = df[df["is_won"] & df["amount"].notna() & df["close_date"].notna()]
    s = won.groupby(won["close_date"].dt.to_period("M"))["amount"].sum()
    last_full = snapshot.to_period("M") - 1
    s = s[s.index <= last_full]
    return s.reindex(pd.period_range(s.index.min(), last_full, freq="M"), fill_value=0.0)


# ------------------------------------------------------------------ models
# Every model has the same signature: fn(history: pd.Series, horizon: int) -> np.ndarray
def predict_naive(y, h):
    """Next months = last month."""
    return np.repeat(y.iloc[-1], h)


def predict_moving_average(y, h):
    """Next months = average of the last 3 months."""
    return np.repeat(y.iloc[-3:].mean(), h)


def predict_seasonal_naive(y, h):
    """Next months = the same month last year."""
    return np.array([y.iloc[-12 + i] for i in range(h)])


def _design_matrix(index: pd.PeriodIndex, origin: pd.Period) -> np.ndarray:
    t = np.array([(p - origin).n for p in index], dtype=float).reshape(-1, 1)
    months = pd.get_dummies(pd.Categorical([p.month for p in index], categories=range(1, 13))).iloc[:, 1:]
    return np.hstack([t, months.to_numpy(dtype=float)])


def predict_trend_seasonal(y, h):
    """Linear regression on a time trend + month-of-year dummies."""
    model = LinearRegression().fit(_design_matrix(y.index, y.index[0]), y.to_numpy())
    future = pd.period_range(y.index[-1] + 1, periods=h, freq="M")
    return np.clip(model.predict(_design_matrix(future, y.index[0])), 0, None)


def predict_ensemble(y, h):
    """Average of moving average, seasonal naive and regression - averaging damps single-model noise."""
    return np.mean([predict_moving_average(y, h), predict_seasonal_naive(y, h), predict_trend_seasonal(y, h)], axis=0)


MODELS = {
    "Naive (last month)": predict_naive,
    "3-month moving average": predict_moving_average,
    "Seasonal naive (same month last year)": predict_seasonal_naive,
    "Trend + seasonality (regression)": predict_trend_seasonal,
    "Ensemble (average of 3 models)": predict_ensemble,
}


# ------------------------------------------------------------------ backtest
def backtest(y: pd.Series, n_test: int = C.BACKTEST_MONTHS):
    """Rolling-origin one-step-ahead backtest. Returns (predictions_df, metrics_df)."""
    rows = []
    for k in range(n_test, 0, -1):
        train = y.iloc[: len(y) - k]          # only data available at forecast time
        actual = y.iloc[len(y) - k]
        for name, fn in MODELS.items():
            rows.append({"period": y.index[len(y) - k], "model": name,
                         "actual": actual, "forecast": float(fn(train, 1)[0])})
    preds = pd.DataFrame(rows)
    preds["error"] = preds["forecast"] - preds["actual"]

    def _metrics(g):
        return pd.Series({
            "MAE": g["error"].abs().mean(),
            "RMSE": np.sqrt((g["error"] ** 2).mean()),
            "WMAPE_pct": 100 * g["error"].abs().sum() / g["actual"].sum(),
            "bias_pct": 100 * g["error"].sum() / g["actual"].sum(),
        })

    metrics = preds.groupby("model").apply(_metrics, include_groups=False).sort_values("WMAPE_pct").reset_index()
    return preds, metrics


# ------------------------------------------------------------------ pipeline view
def weighted_pipeline_by_month(df: pd.DataFrame, periods: pd.PeriodIndex) -> pd.DataFrame:
    """Expected revenue from the ACTIVE open pipeline (stale / overdue deals excluded)."""
    act = df[df["in_active_pipeline"]].copy()
    act["period"] = act["close_date"].dt.to_period("M")
    g = act.groupby("period").agg(open_pipeline=("amount", "sum"), weighted_pipeline=("weighted_amount", "sum"),
                                  open_deals=("opportunity_id", "count"))
    return g.reindex(periods, fill_value=0.0)


# ------------------------------------------------------------------ orchestration
def run_forecast(df: pd.DataFrame, targets: pd.DataFrame, snapshot=C.SNAPSHOT_DATE) -> dict:
    y = monthly_won_revenue(df, snapshot)
    preds, metrics = backtest(y)
    best_row = metrics.iloc[0]
    # Tie-break: with only a handful of test months, small WMAPE gaps are noise. If the ensemble is
    # within ENSEMBLE_TIE_BREAK_PP points of the winner we prefer it because averaging is more robust.
    ens = metrics[metrics["model"].str.startswith("Ensemble")].iloc[0]
    if ens["WMAPE_pct"] <= best_row["WMAPE_pct"] + C.ENSEMBLE_TIE_BREAK_PP:
        best_row = ens
    best = best_row["model"]
    rmse = float(best_row["RMSE"])

    future = pd.period_range(y.index[-1] + 1, periods=C.FORECAST_MONTHS, freq="M")
    fc = MODELS[best](y, C.FORECAST_MONTHS)
    out = pd.DataFrame({"period": future, "statistical_forecast": fc,
                        "lower_80": np.clip(fc - C.INTERVAL_Z * rmse, 0, None),
                        "upper_80": fc + C.INTERVAL_Z * rmse})
    out = out.merge(weighted_pipeline_by_month(df, future).reset_index(names="period"), on="period", how="left")
    out["pipeline_coverage_pct"] = 100 * out["weighted_pipeline"] / out["statistical_forecast"]

    t = targets.copy()
    t["period"] = pd.to_datetime(t["month"]).dt.to_period("M")
    out = out.merge(t[["period", "revenue_target"]], on="period", how="left")
    out["forecast_vs_target_pct"] = 100 * (out["statistical_forecast"] / out["revenue_target"] - 1)
    out["model_used"] = best
    return {"history": y, "backtest_predictions": preds, "backtest_metrics": metrics,
            "best_model": best, "forecast": out}
