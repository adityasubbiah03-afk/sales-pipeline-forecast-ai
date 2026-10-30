"""Central configuration: paths, business rules and model settings.

Everything a business user might want to tweak (stale-deal threshold,
stage probabilities, forecast horizon...) lives here so the rest of the code
contains no "magic numbers".
"""
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
SQL_DIR = ROOT / "sql"

# ---------------------------------------------------------------- reproducibility
SEED = 42

# The date the CRM extract was "taken". Everything (stale deals, overdue deals,
# last full month for forecasting) is measured relative to this date.
SNAPSHOT_DATE = pd.Timestamp("2026-10-01")

# ---------------------------------------------------------------- CRM business rules
CANONICAL_STAGES = ["Prospecting", "Qualification", "Proposal", "Negotiation", "Won", "Lost"]
OPEN_STAGES = ["Prospecting", "Qualification", "Proposal", "Negotiation"]
CLOSED_STAGES = ["Won", "Lost"]

# Probability that a deal in each stage is eventually won. These are the
# assumptions behind the "weighted pipeline". Replace with your own win rates.
STAGE_PROBABILITY = {
    "Prospecting": 0.10,
    "Qualification": 0.20,
    "Proposal": 0.35,
    "Negotiation": 0.50,
}

# An open deal not touched for this many days is considered "stale".
STALE_DAYS = 90

CANONICAL_INDUSTRIES = ["Mining", "Utilities", "Transport", "Government", "Oil & Gas", "Industrial"]

# ---------------------------------------------------------------- forecasting
BACKTEST_MONTHS = 9      # how many recent months are used to test the models
FORECAST_MONTHS = 3      # how many future months to forecast
ENSEMBLE_TIE_BREAK_PP = 1.0  # prefer the ensemble if within this many WMAPE points of the best model
INTERVAL_Z = 1.2816      # z-score for an 80% prediction interval

# ---------------------------------------------------------------- AI commentary
# Model used when ANTHROPIC_API_KEY is set. Override with the ANTHROPIC_MODEL env var.
DEFAULT_LLM_MODEL = "claude-sonnet-5-5"
