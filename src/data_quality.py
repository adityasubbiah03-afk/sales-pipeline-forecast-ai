"""CRM data-quality audit.

Two independent implementations of the same 12 checks:
  * run_pandas_audit  - vectorised pandas
  * run_sql_audit     - plain SQL (sql/data_quality_checks.sql) executed on an in-memory SQLite DB
`reconcile()` compares them, which doubles as a test that the logic is right.
"""
import sqlite3

import pandas as pd

from src import config as C
from src.cleaning import normalise_stage

# check_id: (table, description, severity, recommended fix)
CHECKS = {
    "DQ01": ("opportunities", "Duplicate opportunity IDs (extra rows)", "High",
             "Keep the most recently modified record per ID; add a unique constraint in the CRM export."),
    "DQ02": ("opportunities", "Non-standard stage labels", "Medium",
             "Map to the canonical stage list; replace free-text stage with a picklist."),
    "DQ03": ("opportunities", "Missing deal amount", "High",
             "Exclude from revenue maths; make Amount mandatory beyond the Qualification stage."),
    "DQ04": ("opportunities", "Zero or negative deal amount", "High",
             "Exclude from revenue maths; add a validation rule (Amount > 0)."),
    "DQ05": ("opportunities", "Closed deal with no close date", "High",
             "Impute from last-modified date (flagged); make Close Date mandatory on closure."),
    "DQ06": ("opportunities", "Close date earlier than created date", "High",
             "Treat as invalid, repair as above; add a validation rule (Close >= Created)."),
    "DQ07": ("opportunities", "Deal has no owner", "Medium",
             "Assign to UNASSIGNED and route to sales ops for allocation."),
    "DQ08": ("opportunities", "Deal linked to an account that does not exist (orphan)", "Medium",
             "Map to an 'Unknown' account; find and restore the missing account records."),
    "DQ09": ("opportunities", "Missing lead source", "Low",
             "Set to 'Unknown'; make Lead Source mandatory at creation."),
    "DQ10": ("opportunities", f"Open deal not updated for >{C.STALE_DAYS} days (stale)", "High",
             "Exclude from forecast; reps to re-qualify or close out."),
    "DQ11": ("opportunities", "Open deal whose expected close date has passed (overdue)", "High",
             "Exclude from forecast; reps to update the close date or close out."),
    "DQ12": ("accounts", "Non-standard industry labels", "Low",
             "Standardise to the canonical industry list; use a picklist."),
}


def run_pandas_audit(opps: pd.DataFrame, accounts: pd.DataFrame, snapshot=C.SNAPSHOT_DATE) -> pd.DataFrame:
    o = opps.copy()
    for c in ["created_date", "close_date", "last_modified_date"]:
        o[c] = pd.to_datetime(o[c], errors="coerce")
    norm = o["stage"].map(normalise_stage)
    is_closed = norm.isin(C.CLOSED_STAGES)
    is_open = norm.isin(C.OPEN_STAGES)

    counts = {
        "DQ01": len(o) - o["opportunity_id"].nunique(),
        "DQ02": (~o["stage"].isin(C.CANONICAL_STAGES)).sum(),
        "DQ03": o["amount"].isna().sum(),
        "DQ04": (o["amount"] <= 0).sum(),
        "DQ05": (is_closed & o["close_date"].isna()).sum(),
        "DQ06": (o["close_date"] < o["created_date"]).sum(),
        "DQ07": o["owner_id"].isna().sum(),
        "DQ08": (~o["account_id"].isin(accounts["account_id"])).sum(),
        "DQ09": o["lead_source"].isna().sum(),
        "DQ10": (is_open & (o["last_modified_date"] < snapshot - pd.Timedelta(days=C.STALE_DAYS))).sum(),
        "DQ11": (is_open & (o["close_date"] < snapshot)).sum(),
        "DQ12": (~accounts["industry"].isin(C.CANONICAL_INDUSTRIES)).sum(),
    }
    rows = []
    for cid, (table, desc, sev, fix) in CHECKS.items():
        total = len(accounts) if table == "accounts" else len(o)
        n = int(counts[cid])
        rows.append({"check_id": cid, "table": table, "description": desc, "severity": sev,
                     "rows_affected": n, "pct_of_table": round(100 * n / total, 2), "recommended_fix": fix})
    return pd.DataFrame(rows)


def run_sql_audit(opps_csv, accounts_csv, snapshot=C.SNAPSHOT_DATE) -> pd.DataFrame:
    """Run the SQL version of the checks on the raw CSVs, loaded into in-memory SQLite."""
    con = sqlite3.connect(":memory:")
    # dates stay as ISO text ('YYYY-MM-DD'), which sorts correctly as text in SQLite
    pd.read_csv(opps_csv).to_sql("opportunities_raw", con, index=False)
    pd.read_csv(accounts_csv).to_sql("accounts_raw", con, index=False)
    sql = (C.SQL_DIR / "data_quality_checks.sql").read_text(encoding="utf-8")
    result = pd.read_sql_query(sql, con, params={"snapshot": snapshot.strftime("%Y-%m-%d")})
    con.close()
    return result


def reconcile(pandas_audit: pd.DataFrame, sql_audit: pd.DataFrame) -> list:
    """Return a list of check_ids where pandas and SQL disagree (empty list = full agreement)."""
    merged = pandas_audit.merge(sql_audit, on="check_id", suffixes=("_pandas", "_sql"))
    return merged.loc[merged["rows_affected_pandas"] != merged["rows_affected_sql"], "check_id"].tolist()


def write_report(audit: pd.DataFrame, cleaning_log: dict, path):
    lines = ["# CRM Data-Quality Report", "",
             f"Snapshot date: **{C.SNAPSHOT_DATE:%Y-%m-%d}** | Stale threshold: **{C.STALE_DAYS} days**", "",
             "## Issues found in the raw extract", "",
             "| ID | Table | Issue | Severity | Rows | % of table | Recommended fix |",
             "|----|-------|-------|----------|-----:|-----------:|-----------------|"]
    for r in audit.sort_values("rows_affected", ascending=False).itertuples():
        lines.append(f"| {r.check_id} | {r.table} | {r.description} | {r.severity} | "
                     f"{r.rows_affected:,} | {r.pct_of_table:.2f}% | {r.recommended_fix} |")
    lines += ["", "## What the cleaning step did", "", "| Action | Rows |", "|--------|-----:|"]
    for k, v in cleaning_log.items():
        lines.append(f"| {k.replace('_', ' ').capitalize()} | {v:,} |")
    lines += ["", "Every repaired record is flagged with a `dq_*` column in "
              "`data/processed/fact_opportunities.csv`, so nothing is changed silently.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
