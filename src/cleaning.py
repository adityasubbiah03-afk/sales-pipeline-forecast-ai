"""Cleaning rules shared by the audit and the cleaning step.

Design principle: the *audit* (data_quality.py) only measures problems in the raw
data; the *cleaning* (clean_opportunities) fixes what can safely be fixed and
FLAGS everything it touched with a `dq_*` column, so nothing is changed silently.
"""
import numpy as np
import pandas as pd

from src import config as C


def normalise_stage(value):
    """Map messy CRM stage labels ('closed won ', 'Proposal/Quote'...) to the canonical list."""
    if pd.isna(value):
        return np.nan
    s = str(value).strip().lower()
    if "won" in s:
        return "Won"
    if "lost" in s:
        return "Lost"
    if "prospect" in s:
        return "Prospecting"
    if "qualif" in s:
        return "Qualification"
    if "proposal" in s:
        return "Proposal"
    if "negotiat" in s:
        return "Negotiation"
    return np.nan


def normalise_industry(value):
    """'MINING ', 'Oil and Gas' -> 'Mining', 'Oil & Gas'."""
    if pd.isna(value):
        return "Unknown"
    s = str(value).strip().lower().replace(" and ", " & ")
    lookup = {i.lower(): i for i in C.CANONICAL_INDUSTRIES}
    return lookup.get(s, "Unknown")


def clean_accounts(accounts: pd.DataFrame) -> pd.DataFrame:
    out = accounts.copy()
    out["industry"] = out["industry"].map(normalise_industry)
    return out


def clean_opportunities(opps: pd.DataFrame, snapshot=C.SNAPSHOT_DATE):
    """Return (clean_dataframe, log_dict). The log records how many rows each rule changed."""
    df = opps.copy()
    log = {}

    for col in ["created_date", "close_date", "last_modified_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # 1. Standardise stage labels
    raw_stage = df["stage"].copy()
    df["stage"] = df["stage"].map(normalise_stage)
    log["stage_labels_standardised"] = int((raw_stage != df["stage"]).sum())

    # 2. Remove duplicate opportunity ids, keeping the most recently modified record
    before = len(df)
    df = (
        df.sort_values(["opportunity_id", "last_modified_date"])
        .drop_duplicates("opportunity_id", keep="last")
        .reset_index(drop=True)
    )
    log["duplicate_rows_removed"] = before - len(df)

    is_closed = df["stage"].isin(C.CLOSED_STAGES)
    is_open = df["stage"].isin(C.OPEN_STAGES)

    # 3. Invalid amounts (missing, zero or negative) -> NaN + flag.
    #    We do not guess a value: these deals are simply excluded from revenue maths.
    df["dq_invalid_amount"] = df["amount"].isna() | (df["amount"] <= 0)
    df.loc[df["dq_invalid_amount"], "amount"] = np.nan
    log["invalid_amounts_nulled"] = int(df["dq_invalid_amount"].sum())

    # 4. Bad close dates (missing, or before the created date)
    bad_close = df["close_date"].isna() | (df["close_date"] < df["created_date"])
    # Closed deals: impute from last_modified_date (the last time the rep touched the deal,
    # which for a closed deal is almost always the closing event).
    impute_value = df["last_modified_date"].where(
        df["last_modified_date"] >= df["created_date"], df["created_date"]
    )
    fix_mask = bad_close & is_closed
    df.loc[fix_mask, "close_date"] = impute_value[fix_mask]
    df["dq_close_date_imputed"] = fix_mask
    # Open deals: an invalid expected close date cannot be trusted -> blank it out.
    df.loc[bad_close & is_open, "close_date"] = pd.NaT
    log["closed_deals_close_date_imputed"] = int(fix_mask.sum())
    log["open_deals_close_date_blanked"] = int((bad_close & is_open).sum())

    # 5. Missing owner / lead source
    df["dq_missing_owner"] = df["owner_id"].isna()
    df["owner_id"] = df["owner_id"].fillna("UNASSIGNED")
    df["lead_source"] = df["lead_source"].fillna("Unknown")
    log["missing_owner_assigned"] = int(df["dq_missing_owner"].sum())

    # 6. Open-deal hygiene flags
    df["dq_stale"] = is_open & (df["last_modified_date"] < snapshot - pd.Timedelta(days=C.STALE_DAYS))
    df["dq_overdue"] = is_open & (df["close_date"] < snapshot)
    log["stale_open_deals_flagged"] = int(df["dq_stale"].sum())
    log["overdue_open_deals_flagged"] = int(df["dq_overdue"].sum())

    # 7. Helper columns used by every downstream step
    df["is_open"] = is_open
    df["is_won"] = df["stage"].eq("Won")
    df["is_lost"] = df["stage"].eq("Lost")
    df["stage_probability"] = df["stage"].map(C.STAGE_PROBABILITY)
    # "Active" pipeline = open, trustworthy and still moving. Only this feeds the forecast.
    df["in_active_pipeline"] = (
        is_open & ~df["dq_stale"] & ~df["dq_overdue"] & df["close_date"].notna() & df["amount"].notna()
    )
    df["weighted_amount"] = np.where(is_open, df["amount"] * df["stage_probability"], np.nan)
    df["gross_margin_amount"] = df["amount"] * df["gross_margin_pct"]
    df["sales_cycle_days"] = np.where(
        is_closed, (df["close_date"] - df["created_date"]).dt.days, np.nan
    )
    df["close_month"] = df["close_date"].dt.to_period("M").dt.to_timestamp()
    df["created_month"] = df["created_date"].dt.to_period("M").dt.to_timestamp()

    log["rows_out"] = len(df)
    return df, log
