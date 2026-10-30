"""Static charts (PNG) saved to reports/figures/ - they are embedded in the README."""
import matplotlib

matplotlib.use("Agg")  # no display needed (works in CI)
import matplotlib.pyplot as plt
import pandas as pd

from src import config as C

NAVY, TEAL, ORANGE, GREY, RED = "#1f3a5f", "#2a9d8f", "#e76f51", "#8d99ae", "#c1121f"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                     "axes.titleweight": "bold", "axes.titlesize": 12, "figure.dpi": 130})


def _m(ax, axis="y"):
    fmt = plt.FuncFormatter(lambda v, _: f"${v / 1e6:.1f}M")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def _save(fig, name):
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(C.FIGURES / name, bbox_inches="tight")
    plt.close(fig)


def revenue_vs_target(history, targets):
    t = targets.copy()
    t["period"] = pd.to_datetime(t["month"]).dt.to_period("M")
    t = t.set_index("period")["revenue_target"].reindex(history.index)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.bar(history.index.to_timestamp(), history.values, width=22, color=TEAL, label="Won revenue")
    ax.plot(history.index.to_timestamp(), t.values, color=NAVY, lw=2, marker="o", ms=3, label="Target")
    ax.set_title("Monthly won revenue vs target")
    _m(ax)
    ax.legend(frameon=False)
    _save(fig, "01_revenue_vs_target.png")


def backtest_chart(history, preds, best):
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(history.index.to_timestamp(), history.values, color=GREY, lw=1.5, label="Actual")
    for name, col, ls in [(best, ORANGE, "-"), ("Naive (last month)", NAVY, ":")]:
        g = preds[preds["model"] == name]
        ax.plot(g["period"].dt.to_timestamp(), g["forecast"], color=col, ls=ls, lw=2, marker="o", ms=4,
                label=f"{name} (1-month-ahead)")
    ax.set_title("Backtest: forecasts made only with data available at the time")
    _m(ax)
    ax.legend(frameon=False, fontsize=8)
    _save(fig, "02_backtest.png")


def forecast_chart(history, fc):
    fig, ax = plt.subplots(figsize=(9, 3.8))
    h = history.iloc[-18:]
    ax.plot(h.index.to_timestamp(), h.values, color=GREY, lw=2, label="Actual")
    x = fc["period"].dt.to_timestamp()
    ax.plot(x, fc["statistical_forecast"], color=ORANGE, lw=2, marker="o", label="Statistical forecast")
    ax.fill_between(x, fc["lower_80"], fc["upper_80"], color=ORANGE, alpha=0.18, label="80% interval")
    ax.bar(x, fc["weighted_pipeline"], width=10, color=TEAL, alpha=0.7, label="Weighted active pipeline")
    ax.plot(x, fc["revenue_target"], color=NAVY, ls="--", marker="s", ms=4, label="Target")
    ax.set_title("Next-quarter forecast vs weighted pipeline vs target")
    _m(ax)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    _save(fig, "03_forecast.png")


def pipeline_by_stage(df):
    o = df[df["is_open"] & df["amount"].notna()]
    order = C.OPEN_STAGES
    healthy = o[~(o["dq_stale"] | o["dq_overdue"])].groupby("stage")["amount"].sum().reindex(order, fill_value=0)
    risky = o[o["dq_stale"] | o["dq_overdue"]].groupby("stage")["amount"].sum().reindex(order, fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.barh(order, healthy.values, color=TEAL, label="Active")
    ax.barh(order, risky.values, left=healthy.values, color=RED, alpha=0.75, label="Stale / overdue (excluded from forecast)")
    ax.invert_yaxis()
    ax.set_title("Open pipeline by stage")
    _m(ax, "x")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, "04_pipeline_by_stage.png")


def dq_chart(audit):
    a = audit.sort_values("rows_affected")
    color = a["severity"].map({"High": RED, "Medium": ORANGE, "Low": GREY})
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.barh(a["description"].str.slice(0, 55), a["rows_affected"], color=color)
    ax.set_title("Data-quality issues found in the raw CRM extract (red = high severity)")
    ax.set_xlabel("Rows affected")
    _save(fig, "05_data_quality.png")
