"""Run the whole project end to end:  python -m src.run_pipeline

generate data (if missing) -> audit (pandas + SQL) -> clean -> forecast -> KPIs/anomalies
-> charts -> AI commentary. Outputs land in data/processed/ and reports/.
"""
import argparse
import json

import pandas as pd

from src import config as C
from src import data_quality as dq
from src import forecast as fc
from src import generate_data, kpis, visuals
from src.ai_commentary import generate_commentary
from src.cleaning import clean_accounts, clean_opportunities


def build_dimensions(accounts, reps, products, opps_clean):
    """Add 'Unassigned' / 'Unknown' members so every fact row joins to a dimension (clean star schema)."""
    reps_out = pd.concat([reps, pd.DataFrame([{"rep_id": "UNASSIGNED", "rep_name": "Unassigned",
                                               "team": "Unassigned", "region": "Unknown"}])], ignore_index=True)
    known = set(accounts["account_id"])
    orphan = ~opps_clean["account_id"].isin(known)
    opps_clean["dq_orphan_account"] = orphan
    opps_clean.loc[orphan, "account_id"] = "ACC-UNKNOWN"
    accounts_out = pd.concat([accounts, pd.DataFrame([{"account_id": "ACC-UNKNOWN", "account_name": "Unknown account",
                                                       "industry": "Unknown", "state": "Unknown", "size_band": "Unknown"}])],
                             ignore_index=True)
    return accounts_out, reps_out, orphan.sum()


def build_date_table(start, end):
    d = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})
    d["year"], d["quarter"], d["month_num"] = d.date.dt.year, d.date.dt.quarter, d.date.dt.month
    d["month_name"], d["month_start"] = d.date.dt.strftime("%b"), d.date.dt.to_period("M").dt.to_timestamp()
    d["fin_year_au"] = d.year + (d.month_num >= 7).astype(int)   # FY ends 30 June
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--regenerate", action="store_true", help="re-create the synthetic raw data")
    ap.add_argument("--no-llm", action="store_true", help="skip the LLM and use the rule-based commentary")
    args = ap.parse_args(argv)

    for d in (C.DATA_PROCESSED, C.REPORTS, C.FIGURES):
        d.mkdir(parents=True, exist_ok=True)
    if args.regenerate or not (C.DATA_RAW / "opportunities.csv").exists():
        generate_data.main()

    # ---- load raw
    opps_raw = pd.read_csv(C.DATA_RAW / "opportunities.csv")
    accounts_raw = pd.read_csv(C.DATA_RAW / "accounts.csv")
    reps = pd.read_csv(C.DATA_RAW / "sales_reps.csv")
    products = pd.read_csv(C.DATA_RAW / "products.csv")
    targets = pd.read_csv(C.DATA_RAW / "targets.csv")

    # ---- 1. audit (two independent implementations, reconciled)
    audit = dq.run_pandas_audit(opps_raw, accounts_raw)
    sql_audit = dq.run_sql_audit(C.DATA_RAW / "opportunities.csv", C.DATA_RAW / "accounts.csv")
    mismatches = dq.reconcile(audit, sql_audit)
    print(f"[audit] {len(audit)} checks; SQL vs pandas: "
          f"{'all agree' if not mismatches else 'MISMATCH in ' + ', '.join(mismatches)}")
    audit.to_csv(C.REPORTS / "data_quality_audit.csv", index=False)

    # ---- 2. clean + model-ready tables
    accounts = clean_accounts(accounts_raw)
    opps, log = clean_opportunities(opps_raw)
    accounts_out, reps_out, n_orphans = build_dimensions(accounts, reps, products, opps)
    log["orphan_accounts_remapped"] = int(n_orphans)
    dq.write_report(audit, log, C.REPORTS / "data_quality_report.md")

    date_dim = build_date_table("2023-07-01", "2026-12-31")
    opps.to_csv(C.DATA_PROCESSED / "fact_opportunities.csv", index=False, date_format="%Y-%m-%d")
    accounts_out.to_csv(C.DATA_PROCESSED / "dim_accounts.csv", index=False)
    reps_out.to_csv(C.DATA_PROCESSED / "dim_sales_reps.csv", index=False)
    products.to_csv(C.DATA_PROCESSED / "dim_products.csv", index=False)
    targets.to_csv(C.DATA_PROCESSED / "fact_targets.csv", index=False)
    date_dim.to_csv(C.DATA_PROCESSED / "dim_date.csv", index=False, date_format="%Y-%m-%d")
    print(f"[clean] {len(opps_raw):,} raw rows -> {len(opps):,} clean rows; {len(log)} cleaning rules applied")

    # ---- 3. forecast
    res = fc.run_forecast(opps, targets)
    res["backtest_metrics"].to_csv(C.REPORTS / "forecast_backtest_metrics.csv", index=False)
    res["backtest_predictions"].to_csv(C.REPORTS / "forecast_backtest_predictions.csv", index=False)
    res["forecast"].to_csv(C.REPORTS / "forecast_next_quarter.csv", index=False)
    top = res["backtest_metrics"].iloc[0]
    print(f"[forecast] best raw WMAPE: {top['model']} ({top['WMAPE_pct']:.1f}%); model used: {res['best_model']}")

    # ---- 4. KPIs + anomalies
    summary = kpis.compute_kpis(opps, accounts_out, targets, res["forecast"])
    (C.REPORTS / "kpi_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[kpis] {len(summary['anomalies'])} anomalies flagged")

    # ---- 5. charts
    visuals.revenue_vs_target(res["history"], targets)
    visuals.backtest_chart(res["history"], res["backtest_predictions"], res["best_model"])
    visuals.forecast_chart(res["history"], res["forecast"])
    visuals.pipeline_by_stage(opps)
    visuals.dq_chart(audit)

    # ---- 6. commentary
    text = generate_commentary(summary, use_llm=not args.no_llm)
    (C.REPORTS / "sales_commentary.md").write_text(text, encoding="utf-8")
    print(f"[commentary] written to {C.REPORTS / 'sales_commentary.md'}")
    print("[done] Open reports/ for results and data/processed/ for the Power BI-ready tables.")


if __name__ == "__main__":
    main()
