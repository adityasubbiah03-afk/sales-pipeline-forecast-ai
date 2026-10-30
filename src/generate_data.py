"""Generate a synthetic CRM extract for a two-way-radio / telecoms integrator.

WHY SYNTHETIC? Real CRM data is confidential. This generator produces data with
realistic structure (accounts, reps, products, a sales pipeline with stages) AND
realistic *mess* (duplicates, inconsistent labels, missing values, stale deals) so
the data-quality audit has real work to do. Fixed random seed => fully reproducible.

Tables written to data/raw/:
    accounts.csv, sales_reps.csv, products.csv, opportunities.csv, targets.csv
"""
import numpy as np
import pandas as pd

from src import config as C

# product: (category, typical deal value AUD, typical gross margin)
PRODUCTS = {
    "Handheld Radios": ("Hardware", 45_000, 0.32),
    "Mobile Radios": ("Hardware", 60_000, 0.30),
    "Repeaters & Base Stations": ("Infrastructure", 120_000, 0.28),
    "Network Infrastructure": ("Infrastructure", 260_000, 0.22),
    "Installation Services": ("Services", 80_000, 0.40),
    "Managed Maintenance": ("Services", 150_000, 0.45),
}

INDUSTRY_TEAM = {
    "Mining": "Mining & Resources",
    "Oil & Gas": "Mining & Resources",
    "Utilities": "Utilities & Transport",
    "Transport": "Utilities & Transport",
    "Government": "Government & Industrial",
    "Industrial": "Government & Industrial",
}
INDUSTRY_WIN_ADJ = {"Mining": 0.05, "Oil & Gas": 0.03, "Utilities": 0.02,
                     "Transport": 0.0, "Government": -0.04, "Industrial": -0.01}

REP_NAMES = ["Olivia Chen", "Liam O'Brien", "Priya Nair", "Jack Thompson", "Sofia Rossi",
             "Noah Williams", "Mia Patel", "Ethan Walker", "Chloe Nguyen", "Lucas Martin",
             "Grace Kelly", "Oliver Singh", "Zoe Campbell", "Harry Evans"]
TEAMS = ["Mining & Resources"] * 5 + ["Utilities & Transport"] * 5 + ["Government & Industrial"] * 4
REGIONS = ["VIC", "NSW", "QLD", "WA", "NSW", "VIC", "QLD", "WA", "NSW", "VIC", "NSW", "QLD", "WA", "VIC"]

ADJECTIVES = ["Northern", "Southern", "Pacific", "Summit", "Ridge", "Coastal", "Central", "Apex",
              "Horizon", "Outback", "Granite", "Redback", "Meridian", "Titan", "Evergreen", "Bluestone"]
SUFFIXES = ["Group", "Holdings", "Services", "Operations", "Australia", "Resources"]
NOUNS = {"Mining": ["Iron", "Gold", "Copper", "Coal", "Lithium"], "Oil & Gas": ["Energy", "Petroleum", "Gas"],
         "Utilities": ["Water", "Power", "Grid"], "Transport": ["Rail", "Freight", "Ports", "Logistics"],
         "Government": ["Council", "Emergency Services", "Roads Authority"],
         "Industrial": ["Manufacturing", "Steel", "Engineering"]}

# creation-volume seasonality by calendar month (Australian financial year ends 30 June)
SEASONAL = {1: 0.80, 2: 0.90, 3: 1.05, 4: 1.15, 5: 1.10, 6: 0.95,
            7: 0.85, 8: 0.95, 9: 1.05, 10: 1.15, 11: 1.10, 12: 0.90}

STAGE_VARIANTS = {
    "Won": ["won", "Closed Won", "WON "], "Lost": ["lost", "Closed Lost", " Lost"],
    "Prospecting": ["prospecting", "Prospect"], "Qualification": ["qualification ", "Qualified"],
    "Proposal": ["proposal ", "Proposal/Quote"], "Negotiation": ["negotiation", "Negotiation/Review"],
}
INDUSTRY_VARIANTS = {"Mining": ["mining", "MINING "], "Oil & Gas": ["Oil and Gas", "oil & gas"],
                     "Utilities": [" Utilities", "utilities"], "Transport": ["transport", "TRANSPORT"],
                     "Government": ["government", "GOVERNMENT "], "Industrial": ["industrial", " Industrial"]}


def build_accounts(rng, n=320):
    industries = rng.choice(C.CANONICAL_INDUSTRIES, size=n, p=[0.25, 0.20, 0.15, 0.15, 0.12, 0.13])
    states = rng.choice(["VIC", "NSW", "QLD", "WA", "SA", "NT", "TAS"], size=n,
                        p=[0.20, 0.22, 0.20, 0.25, 0.06, 0.04, 0.03])
    names, seen = [], set()
    for ind in industries:
        while True:
            name = f"{rng.choice(ADJECTIVES)} {rng.choice(NOUNS[ind])} {rng.choice(SUFFIXES)}"
            if name not in seen:
                seen.add(name)
                names.append(name)
                break
    return pd.DataFrame({
        "account_id": [f"ACC-{i:04d}" for i in range(1, n + 1)],
        "account_name": names, "industry": industries, "state": states,
        "size_band": rng.choice(["Small", "Medium", "Large", "Enterprise"], size=n, p=[0.2, 0.35, 0.3, 0.15]),
    })


def build_opportunities(rng, accounts, reps, n=3600):
    snapshot = C.SNAPSHOT_DATE
    # Created dates: seasonal and gently growing over time. Starting mid-2023 means the
    # 2024 closed-deal history is not artificially thin at the start of the window.
    days = pd.date_range("2023-07-01", snapshot - pd.Timedelta(days=7), freq="D")
    w = np.array([SEASONAL[m] for m in days.month]) * (1 + 0.0007 * np.arange(len(days)))
    created = pd.to_datetime(rng.choice(days.values, size=n, p=w / w.sum()))

    acc_idx = rng.integers(0, len(accounts), size=n)
    acc = accounts.iloc[acc_idx].reset_index(drop=True)
    product = rng.choice(list(PRODUCTS), size=n, p=[0.20, 0.17, 0.17, 0.10, 0.18, 0.18])

    # owner = random rep from the team that covers the account's industry
    rep_by_team = {t: reps.loc[reps.team == t, "rep_id"].tolist() for t in reps.team.unique()}
    owner = np.array([rng.choice(rep_by_team[INDUSTRY_TEAM[i]]) for i in acc["industry"]])
    rep_skill = dict(zip(reps.rep_id, rng.normal(0, 0.04, len(reps))))

    base = np.array([PRODUCTS[p][1] for p in product], dtype=float)
    amount = np.round(base * rng.lognormal(0, 0.55, n), -2)
    margin = np.clip(np.array([PRODUCTS[p][2] for p in product]) + rng.normal(0, 0.04, n), 0.05, 0.7)

    cycle = np.clip(rng.lognormal(np.log(55), 0.5, n) * (1 + amount / 400_000 * 0.5), 14, 300).round().astype(int)
    close = created + pd.to_timedelta(cycle, unit="D")

    p_win = np.array([0.32 + INDUSTRY_WIN_ADJ[i] + rep_skill[o] for i, o in zip(acc["industry"], owner)])
    p_win = np.clip(p_win - 0.04 * (amount > 250_000), 0.10, 0.60)
    won = rng.random(n) < p_win

    # ---- stage + last-modified, fully vectorised (a row-by-row loop on pandas objects is very slow)
    created_np = created.values.astype("datetime64[D]")
    close_np = close.values.astype("datetime64[D]")
    snap_np = np.datetime64(snapshot.date(), "D")
    closed_by_date = close_np < snap_np
    forgot = closed_by_date & (rng.random(n) < 0.04)          # "overdue" deals reps never closed out
    resolved = closed_by_date & ~forgot

    frac = (snap_np - created_np).astype(int) / cycle
    open_stage = np.where(frac < 0.25, "Prospecting",
                  np.where(frac < 0.5, "Qualification", np.where(frac < 0.8, "Proposal", "Negotiation")))
    stage = np.where(resolved, np.where(won, "Won", "Lost"),
             np.where(forgot, rng.choice(["Proposal", "Negotiation"], size=n), open_stage)).astype(object)

    td = lambda a: a.astype("timedelta64[D]")
    resolved_mod = np.minimum(close_np + td(rng.integers(0, 6, n)), snap_np - np.timedelta64(1, "D"))
    forgot_mod = close_np - td(rng.integers(0, 30, n))
    lag = np.where(rng.random(n) < 0.10, rng.integers(95, 260, n), rng.integers(0, 40, n))
    open_mod = np.maximum(snap_np - td(lag), created_np + np.timedelta64(1, "D"))
    last_mod = np.where(resolved, resolved_mod, np.where(forgot, forgot_mod, open_mod))

    opp = pd.DataFrame({
        "opportunity_id": [f"OPP-{i:05d}" for i in range(1, n + 1)],
        "account_id": acc["account_id"].values, "owner_id": owner, "product": product,
        "lead_source": rng.choice(["Inbound", "Referral", "Tender", "Outbound", "Partner"], size=n,
                                  p=[0.25, 0.2, 0.2, 0.2, 0.15]),
        "stage": stage, "amount": amount, "gross_margin_pct": margin.round(3),
        "created_date": created, "close_date": close, "last_modified_date": last_mod,
    })
    # keep only deals that closed (or will close) inside the extract window
    keep = ~(opp["stage"].isin(["Won", "Lost"]) & (opp["close_date"] < "2024-01-01"))
    keep &= ~(opp["close_date"] < "2024-01-01")
    opp = opp[keep].reset_index(drop=True)
    opp["opportunity_id"] = [f"OPP-{i:05d}" for i in range(1, len(opp) + 1)]
    return opp


def make_targets(opp):
    """Monthly revenue targets: smoothed actuals x 1.06, extrapolated for future months."""
    won = opp[opp.stage == "Won"].copy()
    monthly = won.groupby(won["close_date"].dt.to_period("M"))["amount"].sum()
    monthly = monthly[monthly.index < C.SNAPSHOT_DATE.to_period("M")]
    smooth = monthly.rolling(3, center=True, min_periods=1).mean()
    growth = monthly.iloc[-12:].sum() / monthly.iloc[-24:-12].sum()
    idx = pd.period_range(monthly.index.min(), "2026-12", freq="M")
    target = {}
    for p in idx:
        target[p] = smooth[p] if p in smooth.index else smooth[p - 12] * growth
    t = pd.Series(target) * 1.06
    return pd.DataFrame({"month": t.index.to_timestamp(), "revenue_target": t.round(-2).values})


def inject_mess(rng, opp, accounts):
    """Corrupt the data the way real CRMs get corrupted. Each issue hits a different set of rows."""
    opp = opp.copy()
    accounts = accounts.copy()
    n = len(opp)
    pool = list(rng.permutation(n))

    def take(frac):
        k = int(n * frac)
        chosen, rest = pool[:k], pool[k:]
        pool[:] = rest
        return chosen

    closed_idx = set(opp.index[opp.stage.isin(["Won", "Lost"])])

    # missing close date on closed deals
    ids = [i for i in take(0.04) if i in closed_idx][:70]
    opp.loc[ids, "close_date"] = pd.NaT
    # close date earlier than created date (typo / wrong year)
    ids = take(0.012)
    opp.loc[ids, "close_date"] = opp.loc[ids, "created_date"] - pd.to_timedelta(rng.integers(1, 40, len(ids)), unit="D")
    # zero / negative / missing amounts
    ids = take(0.008)
    opp.loc[ids, "amount"] = rng.choice([0, -1000, -25000], size=len(ids))
    opp.loc[take(0.015), "amount"] = np.nan
    # missing owner, missing lead source
    opp.loc[take(0.015), "owner_id"] = np.nan
    opp.loc[take(0.03), "lead_source"] = np.nan
    # orphan account ids (account deleted / never created)
    ids = take(0.01)
    opp.loc[ids, "account_id"] = [f"ACC-{9000 + i}" for i in range(len(ids))]
    # inconsistent stage labels
    for i in take(0.07):
        opp.loc[i, "stage"] = rng.choice(STAGE_VARIANTS[opp.loc[i, "stage"]])

    # duplicates: half exact copies, half an OLDER version of the same record
    dup_src = opp.sample(n=90, random_state=C.SEED)
    older = dup_src.iloc[45:].copy()
    older["last_modified_date"] = older["last_modified_date"] - pd.Timedelta(days=30)
    opp = pd.concat([opp, dup_src.iloc[:45], older], ignore_index=True)
    opp = opp.sample(frac=1, random_state=C.SEED).reset_index(drop=True)

    # accounts: inconsistent industry labels
    for i in rng.choice(len(accounts), size=int(len(accounts) * 0.08), replace=False):
        accounts.loc[i, "industry"] = rng.choice(INDUSTRY_VARIANTS[accounts.loc[i, "industry"]])
    return opp, accounts


def main(seed=C.SEED):
    rng = np.random.default_rng(seed)
    C.DATA_RAW.mkdir(parents=True, exist_ok=True)

    accounts = build_accounts(rng)
    reps = pd.DataFrame({"rep_id": [f"REP-{i:02d}" for i in range(1, len(REP_NAMES) + 1)],
                         "rep_name": REP_NAMES, "team": TEAMS, "region": REGIONS})
    products = pd.DataFrame([(p, v[0], v[1]) for p, v in PRODUCTS.items()],
                            columns=["product", "category", "typical_deal_value"])

    opp_clean_truth = build_opportunities(rng, accounts, reps)
    targets = make_targets(opp_clean_truth)
    opp_raw, accounts_raw = inject_mess(rng, opp_clean_truth, accounts)

    for col in ["created_date", "close_date", "last_modified_date"]:
        opp_raw[col] = pd.to_datetime(opp_raw[col]).dt.strftime("%Y-%m-%d")
    targets["month"] = pd.to_datetime(targets["month"]).dt.strftime("%Y-%m-%d")

    accounts_raw.to_csv(C.DATA_RAW / "accounts.csv", index=False)
    reps.to_csv(C.DATA_RAW / "sales_reps.csv", index=False)
    products.to_csv(C.DATA_RAW / "products.csv", index=False)
    opp_raw.to_csv(C.DATA_RAW / "opportunities.csv", index=False)
    targets.to_csv(C.DATA_RAW / "targets.csv", index=False)
    print(f"[generate] {len(opp_raw):,} opportunity rows, {len(accounts_raw)} accounts, "
          f"{len(reps)} reps -> {C.DATA_RAW}")


if __name__ == "__main__":
    main()
