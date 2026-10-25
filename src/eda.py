"""12 exploratory analyses. Each one saves a PNG to figures/ and adds its key
numbers to reports/eda_summary.json so the README can quote real figures.

Run from the project root:  python src/eda.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import (BLUE, FIG_DIR, GRID, INK_2, ORANGE, REPORT_DIR, STAGE_ORDER, apply_style)  # noqa: E402
from src.data_prep import load_master  # noqa: E402

SUMMARY = {}


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name)
    plt.close(fig)
    print("saved", name)


def money(x, _=None):
    return f"${x:,.0f}M"


# 1 ---------------------------------------------------------------------------
def fig01_amount_distribution(df):
    a = df["amount_million_usd"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(a, bins=60, color=BLUE, ax=ax[0])
    ax[0].set(title="Round size is heavily right-skewed", xlabel="Round size (US$ M)", ylabel="Rounds")
    sns.histplot(a, bins=60, log_scale=True, color=BLUE, ax=ax[1])
    ax[1].axvline(a.median(), color=ORANGE, lw=2)
    ax[1].text(a.median() * 1.1, ax[1].get_ylim()[1] * 0.92, f"median ${a.median():.2f}M", color=INK_2)
    ax[1].set(title="The same data on a log scale", xlabel="Round size (US$ M, log scale)", ylabel="")
    ax[1].xaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v:g}"))
    save(fig, "01_round_size_distribution.png")
    SUMMARY["01"] = dict(median=round(a.median(), 2), mean=round(a.mean(), 2), p90=round(a.quantile(.9), 2),
                         max=round(a.max(), 2), skew=round(a.skew(), 2))


# 2 ---------------------------------------------------------------------------
def fig02_yearly(df):
    g = df.groupby("year").agg(total=("amount_million_usd", "sum"), rounds=("round_id", "count"),
                               median=("amount_million_usd", "median"))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    cols = [BLUE] * (len(g) - 1) + ["#9bbfe9"]
    ax[0].bar(g.index.astype(str), g["total"], color=cols)
    ax[0].set(title="Total funding by year (US$ M)", xlabel="", ylabel="")
    ax[0].yaxis.set_major_formatter(mtick.FuncFormatter(money))
    ax[0].text(len(g) - 1, g["total"].iloc[-1], "Q1 only", ha="center", va="bottom", color=INK_2, fontsize=9)
    ax[1].bar(g.index.astype(str), g["rounds"], color=cols)
    ax[1].set(title="Number of rounds by year", xlabel="", ylabel="")
    ax[1].text(len(g) - 1, g["rounds"].iloc[-1], "Q1 only", ha="center", va="bottom", color=INK_2, fontsize=9)
    save(fig, "02_funding_by_year.png")
    SUMMARY["02"] = {str(y): dict(total=round(r.total, 1), rounds=int(r.rounds), median=round(r["median"], 2))
                     for y, r in g.iterrows()}


# 3 ---------------------------------------------------------------------------
def fig03_quarterly(df):
    g = df.groupby(["year", "quarter"]).agg(total=("amount_million_usd", "sum"), rounds=("round_id", "count")).reset_index()
    g["period"] = g["year"].astype(str) + " " + g["quarter"]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(g["period"], g["total"], color=BLUE, lw=2, marker="o", ms=5)
    imax = g["total"].idxmax()
    ax.scatter(g.loc[imax, "period"], g.loc[imax, "total"], color=ORANGE, s=70, zorder=3)
    ax.annotate(f"peak: {g.loc[imax, 'period']}\n${g.loc[imax, 'total']:,.0f}M",
                (g.loc[imax, "period"], g.loc[imax, "total"]), xytext=(12, -26), textcoords="offset points", color=INK_2)
    ax.set(title="Quarterly funding is noisy with no clear long-run trend", xlabel="", ylabel="Total funding (US$ M)")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(money))
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels(g["period"], rotation=60, ha="right", fontsize=8)
    save(fig, "03_quarterly_funding_trend.png")
    SUMMARY["03"] = dict(peak_period=g.loc[imax, "period"], peak_total=round(g.loc[imax, "total"], 1),
                         min_period=g.loc[g["total"].idxmin(), "period"], min_total=round(g["total"].min(), 1))


# 4 ---------------------------------------------------------------------------
def fig04_stage(df):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.boxplot(data=df, x="funding_stage", y="amount_million_usd", order=STAGE_ORDER, color=BLUE,
                fliersize=2, linewidth=1, ax=ax, boxprops=dict(alpha=.85))
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"${v:g}M"))
    ax.set(title="Median round size rises at every funding stage", xlabel="", ylabel="Round size (log scale)")
    med = df.groupby("funding_stage", observed=True)["amount_million_usd"].median()
    ax.set_xticks(range(len(STAGE_ORDER)))
    ax.set_xticklabels([f"{s}\nmedian ${med[s]:.2f}M" for s in STAGE_ORDER], fontsize=8)
    save(fig, "04_round_size_by_stage.png")
    SUMMARY["04"] = {s: dict(rounds=int((df.funding_stage == s).sum()), median=round(med[s], 2)) for s in STAGE_ORDER}


# 5 ---------------------------------------------------------------------------
def fig05_countries(df):
    g = df.groupby("country").agg(total=("amount_million_usd", "sum"), rounds=("round_id", "count"))
    top = g.sort_values("total", ascending=False).head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = [ORANGE if c == top.index[-1] else BLUE for c in top.index]
    ax.barh(top.index, top["total"], color=colors)
    for y, (t, r) in enumerate(zip(top["total"], top["rounds"])):
        ax.text(t, y, f"  ${t:,.0f}M  ({r} rounds)", va="center", fontsize=8.5, color=INK_2)
    ax.set(title="Top 15 countries by total funding", xlabel="", ylabel="")
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(money))
    ax.set_xlim(0, top["total"].max() * 1.28)
    ax.grid(axis="y", visible=False)
    save(fig, "05_top_countries.png")
    share = g["total"] / g["total"].sum() * 100
    SUMMARY["05"] = dict(top5=[dict(country=c, total=round(g.loc[c, "total"], 1), rounds=int(g.loc[c, "rounds"]),
                                    share_pct=round(share[c], 1)) for c in g.sort_values("total", ascending=False).index[:5]],
                         top5_share_pct=round(share.sort_values(ascending=False).head(5).sum(), 1))


# 6 ---------------------------------------------------------------------------
def fig06_regions(df):
    g = df.groupby("region").agg(total=("amount_million_usd", "sum"), rounds=("round_id", "count"),
                                 median=("amount_million_usd", "median")).sort_values("total")
    g["share"] = g["total"] / g["total"].sum() * 100
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.barh(g.index, g["share"], color=BLUE)
    for y, (s, t) in enumerate(zip(g["share"], g["total"])):
        ax.text(s, y, f"  {s:.1f}%  (${t:,.0f}M)", va="center", fontsize=9, color=INK_2)
    ax.set(title="Share of total funding by region", xlabel="", ylabel="")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.set_xlim(0, g["share"].max() * 1.3)
    ax.grid(axis="y", visible=False)
    save(fig, "06_funding_share_by_region.png")
    SUMMARY["06"] = {r: dict(share_pct=round(row.share, 1), total=round(row.total, 1), rounds=int(row.rounds)) for r, row in g.iterrows()}


# 7 ---------------------------------------------------------------------------
def fig07_sectors(df):
    g = df.groupby("sector").agg(total=("amount_million_usd", "sum"), rounds=("round_id", "count"),
                                 median=("amount_million_usd", "median")).sort_values("total")
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = [ORANGE if s == g.index[-1] else BLUE for s in g.index]
    ax.barh(g.index, g["total"], color=colors)
    for y, (t, r) in enumerate(zip(g["total"], g["rounds"])):
        ax.text(t, y, f"  ${t:,.0f}M ({r})", va="center", fontsize=8, color=INK_2)
    ax.set(title="Total funding by sector (number of rounds in brackets)", xlabel="", ylabel="")
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(money))
    ax.set_xlim(0, g["total"].max() * 1.2)
    ax.grid(axis="y", visible=False)
    save(fig, "07_funding_by_sector.png")
    top = g.sort_values("total", ascending=False)
    SUMMARY["07"] = dict(top5=[dict(sector=s, total=round(top.loc[s, "total"], 1), rounds=int(top.loc[s, "rounds"])) for s in top.index[:5]],
                         most_rounds=g["rounds"].idxmax(), most_rounds_n=int(g["rounds"].max()))


# 8 ---------------------------------------------------------------------------
def fig08_country_sector_heatmap(df):
    top_c = df.groupby("country")["amount_million_usd"].sum().nlargest(10).index
    top_s = df.groupby("sector")["amount_million_usd"].sum().nlargest(10).index
    p = (df[df.country.isin(top_c) & df.sector.isin(top_s)]
         .pivot_table(index="country", columns="sector", values="amount_million_usd", aggfunc="sum", fill_value=0))
    p = p.loc[top_c, top_s]
    fig, ax = plt.subplots(figsize=(11, 5.8))
    sns.heatmap(p, annot=True, fmt=".0f", cmap="Blues", linewidths=2, linecolor="#fcfcfb",
                cbar_kws=dict(label="Total funding (US$ M)"), annot_kws=dict(fontsize=8), ax=ax)
    ax.set(title="Where the money goes: top 10 countries x top 10 sectors (US$ M)", xlabel="", ylabel="")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    save(fig, "08_country_sector_heatmap.png")
    flat = p.stack()
    SUMMARY["08"] = dict(largest_cell=dict(country=flat.idxmax()[0], sector=flat.idxmax()[1], total=round(flat.max(), 1)))


# 9 ---------------------------------------------------------------------------
def fig09_valuations(df):
    g = df.groupby("funding_stage", observed=True)[["pre_money_valuation_million_usd", "post_money_valuation_million_usd"]].median().loc[STAGE_ORDER]
    x = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.bar(x - .2, g.iloc[:, 0], .38, color="#9bbfe9", label="Pre-money (median)")
    ax.bar(x + .2, g.iloc[:, 1], .38, color=BLUE, label="Post-money (median)")
    ax.set_xticks(x)
    ax.set_xticklabels(STAGE_ORDER)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(money))
    ax.set(title="Median valuation climbs steeply with each stage", xlabel="", ylabel="")
    ax.legend(frameon=False, loc="upper left")
    for xi, v in zip(x + .2, g.iloc[:, 1]):
        ax.text(xi, v, f"${v:,.0f}M", ha="center", va="bottom", fontsize=8.5, color=INK_2)
    save(fig, "09_valuation_by_stage.png")
    SUMMARY["09"] = {s: dict(pre=round(g.loc[s].iloc[0], 1), post=round(g.loc[s].iloc[1], 1)) for s in STAGE_ORDER}


# 10 --------------------------------------------------------------------------
def fig10_investor_type(df):
    g = df.groupby("investor_type").agg(rounds=("round_id", "count"), median=("amount_million_usd", "median"),
                                        total=("amount_million_usd", "sum")).sort_values("rounds")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    ax[0].barh(g.index, g["rounds"], color=BLUE)
    ax[0].set(title="Rounds led, by investor type", xlabel="", ylabel="")
    for y, v in enumerate(g["rounds"]):
        ax[0].text(v, y, f" {v}", va="center", fontsize=9, color=INK_2)
    ax[0].set_xlim(0, g["rounds"].max() * 1.12)
    ax[1].barh(g.index, g["median"], color=BLUE)
    ax[1].set(title="Median round size (US$ M)", xlabel="", ylabel="")
    for y, v in enumerate(g["median"]):
        ax[1].text(v, y, f" ${v:.2f}M", va="center", fontsize=9, color=INK_2)
    ax[1].set_xlim(0, g["median"].max() * 1.25)
    for a in ax:
        a.grid(axis="y", visible=False)
    save(fig, "10_investor_type.png")
    SUMMARY["10"] = {t: dict(rounds=int(r.rounds), median=round(r["median"], 2), total=round(r.total, 1)) for t, r in g.iterrows()}


# 11 --------------------------------------------------------------------------
def fig11_correlation(df):
    cols = {"amount_million_usd": "Round size", "pre_money_valuation_million_usd": "Pre-money val.",
            "post_money_valuation_million_usd": "Post-money val.", "equity_dilution_pct": "Equity dilution %",
            "num_investors": "No. of investors", "employee_count": "Employees",
            "age_at_round": "Startup age at round", "round_number": "Round number", "female_founder": "Female founder"}
    c = df[list(cols)].rename(columns=cols).corr(method="spearman")
    fig, ax = plt.subplots(figsize=(8.5, 6.8))
    cmap = sns.diverging_palette(20, 250, s=80, l=50, center="light", as_cmap=True)
    sns.heatmap(c, annot=True, fmt=".2f", cmap=cmap, vmin=-1, vmax=1, center=0, linewidths=2, linecolor="#fcfcfb",
                annot_kws=dict(fontsize=8), cbar_kws=dict(label="Spearman correlation"), ax=ax)
    ax.set_title("Round size tracks valuation, not startup traits")
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    save(fig, "11_correlation_heatmap.png")
    r = c.loc["Round size"].drop("Round size")
    SUMMARY["11"] = dict(spearman_with_round_size={k: round(v, 3) for k, v in r.items()})


# 12 --------------------------------------------------------------------------
def fig12_startup_traits(df):
    fig, ax = plt.subplots(1, 3, figsize=(12, 4.4), sharey=True)
    order = ["Pre-Revenue", "Revenue Generating", "Profitable"]
    sns.boxplot(data=df, x="revenue_status", y="amount_million_usd", order=order, color=BLUE, fliersize=1.5, linewidth=1, ax=ax[0])
    ax[0].set(title="By revenue status", xlabel="", ylabel="Round size (log scale)")
    df2 = df.assign(founder=np.where(df.female_founder == 1, "Female founder", "No female founder"))
    sns.boxplot(data=df2, x="founder", y="amount_million_usd", order=["No female founder", "Female founder"], color=BLUE, fliersize=1.5, linewidth=1, ax=ax[1])
    ax[1].set(title="By founder gender", xlabel="", ylabel="")
    bins = pd.cut(df["employee_count"], [0, 10, 25, 50, 100, 2000], labels=["1-10", "11-25", "26-50", "51-100", "100+"])
    sns.boxplot(x=bins, y=df["amount_million_usd"], color=BLUE, fliersize=1.5, linewidth=1, ax=ax[2])
    ax[2].set(title="By team size", xlabel="Employees", ylabel="")
    ax[0].set_yscale("log")
    ax[0].yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"${v:g}M"))
    for a in ax:
        a.tick_params(axis="x", labelsize=8.5)
    fig.suptitle("Startup traits barely separate round sizes (all stages mixed)", x=0.01, ha="left", fontweight="bold", fontsize=13)
    fig.tight_layout()
    save(fig, "12_startup_traits.png")
    SUMMARY["12"] = dict(
        median_by_revenue={k: round(v, 2) for k, v in df.groupby("revenue_status")["amount_million_usd"].median().items()},
        median_by_female_founder={("female" if k == 1 else "no_female"): round(v, 2) for k, v in df.groupby("female_founder")["amount_million_usd"].median().items()},
        share_female_founder_rounds_pct=round(df.female_founder.mean() * 100, 1))


def main():
    apply_style()
    df = load_master()
    for fn in [fig01_amount_distribution, fig02_yearly, fig03_quarterly, fig04_stage, fig05_countries, fig06_regions,
               fig07_sectors, fig08_country_sector_heatmap, fig09_valuations, fig10_investor_type, fig11_correlation,
               fig12_startup_traits]:
        fn(df)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "eda_summary.json").write_text(json.dumps(SUMMARY, indent=2, default=str))
    print("12 figures saved to", FIG_DIR)


if __name__ == "__main__":
    main()
