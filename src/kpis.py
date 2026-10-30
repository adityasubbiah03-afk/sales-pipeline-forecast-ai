"""Compute the KPI summary and detect anomalies.

IMPORTANT DESIGN CHOICE: every number and every anomaly is computed here, deterministically,
in Python. The LLM in ai_commentary.py is only allowed to *narrate* this summary - it never
calculates anything, which removes the main risk of AI-generated reporting (invented numbers).
"""
import numpy as np
import pandas as pd

from src import config as C


def _win_rate(closed: pd.DataFrame):
    n = len(closed)
    return float(closed["is_won"].sum() / n) if n else float("nan")


def _window(df, start: pd.Period, end: pd.Period):
    p = df["close_date"].dt.to_period("M")
    return df[(p >= start) & (p <= end) & (df["is_won"] | df["is_lost"])]


def compute_kpis(df: pd.DataFrame, accounts: pd.DataFrame, targets: pd.DataFrame, forecast: pd.DataFrame,
                 snapshot=C.SNAPSHOT_DATE) -> dict:
    last = snapshot.to_period("M") - 1
    cur = _window(df, last - 2, last)           # last 3 full months
    prev = _window(df, last - 5, last - 3)      # the 3 months before that

    won = df[df["is_won"] & df["amount"].notna() & df["close_date"].notna()]
    rev_by_month = won.groupby(won["close_date"].dt.to_period("M"))["amount"].sum()
    rev_by_month = rev_by_month[rev_by_month.index <= last]

    t = targets.copy()
    t["period"] = pd.to_datetime(t["month"]).dt.to_period("M")
    target_last = float(t.loc[t["period"] == last, "revenue_target"].iloc[0])

    def money(x):
        return round(float(x), 0)

    cur_won, prev_won = cur[cur["is_won"] & cur["amount"].notna()], prev[prev["is_won"] & prev["amount"].notna()]
    wr_cur, wr_prev = _win_rate(cur), _win_rate(prev)

    # --- open pipeline
    opn = df[df["is_open"]]
    active = df[df["in_active_pipeline"]]
    open_amt = float(opn["amount"].sum())
    at_risk_amt = float(opn.loc[opn["dq_stale"] | opn["dq_overdue"], "amount"].sum())

    # value-based win rate over the last 12 months, vs the win rate the stage probabilities imply
    last12 = _window(df, last - 11, last)
    last12 = last12[last12["amount"].notna()]
    hist_value_wr = float(last12.loc[last12["is_won"], "amount"].sum() / last12["amount"].sum())
    implied_wr = float(active["weighted_amount"].sum() / active["amount"].sum()) if len(active) else float("nan")

    # --- mix
    def by(col, frame):
        return frame.groupby(col)["amount"].sum().sort_values(ascending=False)

    ind = df.merge(accounts[["account_id", "industry"]], on="account_id", how="left")
    ind_cur = _window(ind, last - 2, last)
    # rank real industries only: drop the 'Unknown' bucket (orphan accounts) and tiny samples
    ind_cur = ind_cur[ind_cur["industry"] != "Unknown"]
    grp = ind_cur.groupby("industry")["is_won"]
    wr_by_ind = grp.mean()[grp.size() >= 10].sort_values()
    prod_cur, prod_prev = by("product", cur_won), by("product", prev_won)
    prod_chg = ((prod_cur / prod_prev - 1) * 100).dropna().sort_values()

    rep_open = opn.groupby("owner_id")["amount"].sum().sort_values(ascending=False)

    gm_cur = float(cur_won["gross_margin_amount"].sum() / cur_won["amount"].sum())
    gm_prev = float(prev_won["gross_margin_amount"].sum() / prev_won["amount"].sum())

    # --- anomalies (rule-based, transparent thresholds)
    anomalies = []
    trailing = rev_by_month.iloc[-13:-1]
    z = float((rev_by_month.iloc[-1] - trailing.mean()) / trailing.std(ddof=1))
    if abs(z) >= 1.5:
        anomalies.append({"type": "revenue_vs_trailing_12m", "detail":
                          f"Last month's won revenue is {z:+.1f} standard deviations from the trailing 12-month average."})
    if (wr_cur - wr_prev) * 100 <= -5:
        anomalies.append({"type": "win_rate_drop", "detail":
                          f"Win rate fell {abs(wr_cur - wr_prev) * 100:.1f} percentage points versus the prior 3 months."})
    if at_risk_amt / open_amt > 0.15:
        anomalies.append({"type": "pipeline_hygiene", "detail":
                          f"{at_risk_amt / open_amt:.0%} of open pipeline value is stale or overdue and excluded from the forecast."})
    if implied_wr - hist_value_wr > 0.10:
        anomalies.append({"type": "pipeline_optimism", "detail":
                          f"Stage probabilities imply a {implied_wr:.0%} win rate but the last 12 months delivered {hist_value_wr:.0%}."})
    for prod, chg in prod_chg.items():
        if chg <= -25:
            anomalies.append({"type": "product_decline", "detail": f"{prod} won revenue is down {abs(chg):.0f}% versus the prior 3 months."})
    if rep_open.iloc[0] / rep_open.sum() > 0.20:
        anomalies.append({"type": "rep_concentration", "detail":
                          f"{rep_open.index[0]} holds {rep_open.iloc[0] / rep_open.sum():.0%} of open pipeline value."})

    f0 = forecast.iloc[0]
    return {
        "as_of": snapshot.strftime("%Y-%m-%d"),
        "last_full_month": str(last),
        "revenue": {
            "last_month_won": money(rev_by_month.iloc[-1]), "last_month_target": money(target_last),
            "last_month_vs_target_pct": round(100 * (rev_by_month.iloc[-1] / target_last - 1), 1),
            "last_3m_won": money(cur_won["amount"].sum()), "prior_3m_won": money(prev_won["amount"].sum()),
            "last_3m_change_pct": round(100 * (cur_won["amount"].sum() / prev_won["amount"].sum() - 1), 1),
        },
        "win_rate": {"last_3m_pct": round(100 * wr_cur, 1), "prior_3m_pct": round(100 * wr_prev, 1),
                     "lowest_industry_last_3m": {"industry": wr_by_ind.index[0], "pct": round(100 * wr_by_ind.iloc[0], 1)},
                     "highest_industry_last_3m": {"industry": wr_by_ind.index[-1], "pct": round(100 * wr_by_ind.iloc[-1], 1)}},
        "sales_cycle_days_median_won": {"last_3m": float(cur_won["sales_cycle_days"].median()),
                                         "prior_3m": float(prev_won["sales_cycle_days"].median())},
        "gross_margin_pct": {"last_3m": round(100 * gm_cur, 1), "prior_3m": round(100 * gm_prev, 1)},
        "pipeline": {
            "open_deals": int(len(opn)), "open_value": money(open_amt),
            "active_weighted_value": money(active["weighted_amount"].sum()),
            "stale_or_overdue_value": money(at_risk_amt), "stale_or_overdue_pct": round(100 * at_risk_amt / open_amt, 1),
            "implied_win_rate_pct": round(100 * implied_wr, 1), "historical_value_win_rate_pct": round(100 * hist_value_wr, 1),
        },
        "top_products_last_3m": {k: money(v) for k, v in prod_cur.head(3).items()},
        "forecast_next_month": {"period": str(f0["period"]), "statistical_forecast": money(f0["statistical_forecast"]),
                                "range_80": [money(f0["lower_80"]), money(f0["upper_80"])],
                                "weighted_pipeline": money(f0["weighted_pipeline"]),
                                "target": money(f0["revenue_target"]), "model": f0["model_used"]},
        "anomalies": anomalies,
    }
