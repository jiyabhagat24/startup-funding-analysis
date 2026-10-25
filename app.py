"""Interactive dashboard: explore global startup funding by country and industry.

Run from the project root:  streamlit run app.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.config import BLUE, FIG_DIR, INK_2, ORANGE, STAGE_ORDER, apply_style  # noqa: E402
from src.data_prep import load_master  # noqa: E402
from src.model import load_or_train, predict_proba_high  # noqa: E402

st.set_page_config(page_title="Global Startup Funding", page_icon=":bar_chart:", layout="wide")
apply_style()

METRICS = {
    "Total funding (US$ M)": ("amount_million_usd", "sum"),
    "Number of rounds": ("round_id", "count"),
    "Median round size (US$ M)": ("amount_million_usd", "median"),
}


# ----------------------------------------------------------------- data ---
@st.cache_data
def get_data():
    return load_master()


@st.cache_resource
def get_model():
    return load_or_train()


df_all = get_data()

# -------------------------------------------------------------- sidebar ---
st.sidebar.header("Filters")
y_min, y_max = int(df_all["year"].min()), int(df_all["year"].max())
years = st.sidebar.slider("Year", y_min, y_max, (y_min, y_max))
regions = st.sidebar.multiselect("Region (empty = all)", sorted(df_all["region"].unique()))
pool = df_all[df_all["region"].isin(regions)] if regions else df_all
countries = st.sidebar.multiselect("Country (empty = all)", sorted(pool["country"].unique()))
sectors = st.sidebar.multiselect("Industry / sector (empty = all)", sorted(df_all["sector"].unique()))
stages = st.sidebar.multiselect("Funding stage (empty = all)", STAGE_ORDER)
top_n = st.sidebar.slider("Show top N", 5, 20, 10)
st.sidebar.caption("2026 contains Q1 only (data runs 2020-01-02 to 2026-03-29).")

mask = df_all["year"].between(*years)
if regions:
    mask &= df_all["region"].isin(regions)
if countries:
    mask &= df_all["country"].isin(countries)
if sectors:
    mask &= df_all["sector"].isin(sectors)
if stages:
    mask &= df_all["funding_stage"].astype(str).isin(stages)
df = df_all[mask].copy()


def fmt_money(x):
    return f"${x:,.0f}M" if x >= 100 else f"${x:,.2f}M"


def money_axis(ax, axis="y"):
    f = mtick.FuncFormatter(lambda v, _: f"${v:,.0f}M")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(f)


def aggregate(frame, by, metric):
    col, fn = METRICS[metric]
    return frame.groupby(by, observed=True)[col].agg(fn)


def bar_top(series, title, metric):
    s = series.sort_values(ascending=False).head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.42 * len(s) + 1)))
    ax.barh(s.index.astype(str), s.values, color=[ORANGE if i == len(s) - 1 else BLUE for i in range(len(s))])
    for y, v in enumerate(s.values):
        ax.text(v, y, f"  {v:,.0f}" if "Number" in metric else f"  ${v:,.2f}M" if v < 100 else f"  ${v:,.0f}M",
                va="center", fontsize=8.5, color=INK_2)
    ax.set(title=title, xlabel=metric, ylabel="")
    ax.set_xlim(0, s.max() * 1.18)
    ax.grid(axis="y", visible=False)
    return fig


def summary_table(frame, by):
    t = frame.groupby(by, observed=True).agg(
        rounds=("round_id", "count"), startups=("startup_id", "nunique"),
        total_funding_million_usd=("amount_million_usd", "sum"),
        median_round_million_usd=("amount_million_usd", "median"),
        max_round_million_usd=("amount_million_usd", "max"))
    return t.sort_values("total_funding_million_usd", ascending=False).round(2)


# ----------------------------------------------------------------- body ---
st.title("Global Startup Funding Explorer")
st.caption(f"{len(df_all):,} funding rounds from {df_all['funding_date'].min():%b %Y} to {df_all['funding_date'].max():%b %Y}, "
           f"{df_all['startup_id'].nunique():,} startups, {df_all['country'].nunique()} countries, {df_all['sector'].nunique()} sectors. "
           "Use the sidebar to filter.")

if df.empty:
    st.warning("No funding rounds match these filters. Widen the filters in the sidebar.")
    st.stop()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total funding", fmt_money(df["amount_million_usd"].sum()))
c2.metric("Funding rounds", f"{len(df):,}")
c3.metric("Startups funded", f"{df['startup_id'].nunique():,}")
c4.metric("Median round", fmt_money(df["amount_million_usd"].median()))
c5.metric("Largest round", fmt_money(df["amount_million_usd"].max()))

tab_trend, tab_country, tab_industry, tab_cross, tab_stage, tab_predict, tab_data = st.tabs(
    ["Trends", "By country", "By industry", "Country x industry", "Stages & investors", "Predict", "Data"])

# ---- Trends
with tab_trend:
    g = df.groupby(["year", "quarter"]).agg(total=("amount_million_usd", "sum"), rounds=("round_id", "count")).reset_index()
    g["period"] = g["year"].astype(str) + " " + g["quarter"]
    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(7, 3.8))
        ax.plot(g["period"], g["total"], color=BLUE, lw=2, marker="o", ms=4)
        ax.set(title="Total funding per quarter", xlabel="", ylabel="")
        money_axis(ax)
        step = max(1, len(g) // 12)
        ax.set_xticks(range(0, len(g), step))
        ax.set_xticklabels(g["period"].iloc[::step], rotation=60, ha="right", fontsize=8)
        st.pyplot(fig)
        plt.close(fig)
    with right:
        y = df.groupby("year").agg(total=("amount_million_usd", "sum"))
        fig, ax = plt.subplots(figsize=(7, 3.8))
        ax.bar(y.index.astype(str), y["total"], color=[BLUE if yr < 2026 else "#9bbfe9" for yr in y.index])
        ax.set(title="Total funding per year (2026 = Q1 only)", xlabel="", ylabel="")
        money_axis(ax)
        st.pyplot(fig)
        plt.close(fig)
    top_sectors = df.groupby("sector")["amount_million_usd"].sum().nlargest(min(top_n, 8)).index
    p = (df[df["sector"].isin(top_sectors)].pivot_table(index="year", columns="sector", values="amount_million_usd", aggfunc="sum", fill_value=0))
    fig, ax = plt.subplots(figsize=(12, 3.8))
    sns.heatmap(p.T, annot=True, fmt=".0f", cmap="Blues", linewidths=2, linecolor="#fcfcfb", cbar_kws=dict(label="US$ M"), annot_kws=dict(fontsize=8), ax=ax)
    ax.set(title="Total funding by year for the largest sectors in this selection (US$ M)", xlabel="", ylabel="")
    st.pyplot(fig)
    plt.close(fig)

# ---- By country
with tab_country:
    metric = st.selectbox("Rank countries by", list(METRICS), key="m_country")
    left, right = st.columns([3, 2])
    with left:
        fig = bar_top(aggregate(df, "country", metric), f"Top {top_n} countries", metric)
        st.pyplot(fig)
        plt.close(fig)
    with right:
        reg = df.groupby("region")["amount_million_usd"].sum().sort_values()
        fig, ax = plt.subplots(figsize=(5, 3.6))
        ax.barh(reg.index, reg.values / reg.sum() * 100, color=BLUE)
        ax.set(title="Share of funding by region", xlabel="% of selected funding", ylabel="")
        ax.grid(axis="y", visible=False)
        st.pyplot(fig)
        plt.close(fig)
    st.subheader("Country table")
    st.dataframe(summary_table(df, "country"))

# ---- By industry
with tab_industry:
    metric = st.selectbox("Rank industries by", list(METRICS), key="m_sector")
    left, right = st.columns([3, 2])
    with left:
        fig = bar_top(aggregate(df, "sector", metric), f"Top {top_n} industries", metric)
        st.pyplot(fig)
        plt.close(fig)
    with right:
        med = df.groupby("sector")["amount_million_usd"].median().sort_values(ascending=False).head(top_n).iloc[::-1]
        fig, ax = plt.subplots(figsize=(5, max(3, 0.4 * len(med) + 1)))
        ax.barh(med.index, med.values, color=BLUE)
        ax.set(title="Median round size (mixes all stages)", xlabel="US$ M", ylabel="")
        ax.grid(axis="y", visible=False)
        st.pyplot(fig)
        plt.close(fig)
    st.subheader("Industry table")
    st.dataframe(summary_table(df, "sector"))

# ---- Country x industry
with tab_cross:
    metric = st.selectbox("Cell value", list(METRICS), key="m_cross")
    col, fn = METRICS[metric]
    top_c = df.groupby("country")["amount_million_usd"].sum().nlargest(top_n).index
    top_s = df.groupby("sector")["amount_million_usd"].sum().nlargest(top_n).index
    sub = df[df["country"].isin(top_c) & df["sector"].isin(top_s)]
    if sub.empty:
        st.info("Not enough data for a heatmap with these filters.")
    else:
        p = sub.pivot_table(index="country", columns="sector", values=col, aggfunc=fn).reindex(index=top_c, columns=top_s)
        fig, ax = plt.subplots(figsize=(12, max(3.5, 0.5 * len(p) + 2)))
        sns.heatmap(p, annot=True, fmt=".0f" if "Median" not in metric else ".1f", cmap="Blues", linewidths=2, linecolor="#fcfcfb",
                    cbar_kws=dict(label=metric), annot_kws=dict(fontsize=8), ax=ax)
        ax.set(title=f"{metric}: top {top_n} countries x top {top_n} industries (blank = no rounds)", xlabel="", ylabel="")
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Countries and industries are ranked by total funding within the current filters.")

# ---- Stages & investors
with tab_stage:
    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        present = [s for s in STAGE_ORDER if s in set(df["funding_stage"].astype(str))]
        sns.boxplot(data=df.assign(funding_stage=df["funding_stage"].astype(str)), x="funding_stage", y="amount_million_usd", order=present,
                    color=BLUE, fliersize=2, linewidth=1, ax=ax)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"${v:g}M"))
        ax.set(title="Round size by funding stage (log scale)", xlabel="", ylabel="")
        plt.setp(ax.get_xticklabels(), rotation=20)
        st.pyplot(fig)
        plt.close(fig)
    with right:
        it = df.groupby("investor_type").agg(rounds=("round_id", "count")).sort_values("rounds")
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.barh(it.index, it["rounds"], color=BLUE)
        ax.set(title="Rounds by investor type", xlabel="Rounds", ylabel="")
        ax.grid(axis="y", visible=False)
        st.pyplot(fig)
        plt.close(fig)
    lead = (df.groupby("lead_investor").agg(rounds=("round_id", "count"), total_funding_million_usd=("amount_million_usd", "sum"))
            .sort_values("total_funding_million_usd", ascending=False).head(top_n).round(2))
    st.subheader(f"Top {top_n} lead investors by funding led")
    st.dataframe(lead)

# ---- Predict
with tab_predict:
    model, info = get_model()
    st.subheader("Funding-potential model (Random Forest)")
    st.write(f"**Question:** will a round be a *high-funding* round, i.e. larger than the median round "
             f"(about **US${info['threshold_million_usd']:.2f}M**)?")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test ROC AUC", f"{info['test']['roc_auc']:.3f}")
    m2.metric("Test accuracy", f"{info['test']['accuracy']:.1%}")
    m3.metric("Stage-only AUC", f"{info['baselines']['stage_only_random_forest']['roc_auc']:.3f}")
    m4.metric("Always-guess-majority accuracy", f"{info['baselines']['majority_class_accuracy']:.1%}")
    st.caption(f"Evaluated on {info['n_test']} held-out rounds from {info['n_startups_test']} startups the model never saw in training. "
               "Funding stage drives almost all of the predictive power; the other features add very little. "
               f"For a round that is large *for its own stage*, the same model scores AUC {info['stage_relative_experiment']['test_roc_auc']:.2f} (about chance).")
    img = FIG_DIR / "model_feature_importance.png"
    if img.exists():
        st.image(str(img))

    st.markdown("#### Try it")
    country_region = df_all.drop_duplicates("country").set_index("country")["region"].to_dict()
    with st.form("predict_form"):
        a, b, c = st.columns(3)
        stage = a.selectbox("Funding stage", STAGE_ORDER, index=1)
        sector = b.selectbox("Industry", sorted(df_all["sector"].unique()))
        country = c.selectbox("Country", sorted(df_all["country"].unique()))
        d, e, f = st.columns(3)
        revenue = d.selectbox("Revenue status", sorted(df_all["revenue_status"].unique()))
        employees = e.number_input("Employees", min_value=1, max_value=2000, value=20)
        age = f.number_input("Startup age (years)", min_value=0, max_value=15, value=2)
        g1, g2 = st.columns(2)
        female = g1.selectbox("Female founder", ["No", "Yes"])
        year = g2.number_input("Funding year", min_value=2020, max_value=2030, value=2025)
        go = st.form_submit_button("Predict")
    if go:
        prob = predict_proba_high(model, funding_stage=stage, sector=sector, country=country, region=country_region[country],
                                  revenue_status=revenue, female_founder=int(female == "Yes"), employee_count=int(employees),
                                  age_at_round=int(age), year=int(year))
        st.metric("Probability of a high-funding round", f"{prob:.0%}")
        st.progress(min(max(prob, 0.0), 1.0))
        st.caption(f"'High' means above US${info['threshold_million_usd']:.2f}M. This is a statistical pattern in this dataset, not financial advice.")

# ---- Data
with tab_data:
    st.write(f"{len(df):,} rounds match the current filters.")
    show = ["round_id", "startup_name", "funding_date", "funding_stage", "amount_million_usd", "pre_money_valuation_million_usd",
            "post_money_valuation_million_usd", "lead_investor", "investor_type", "sector", "country", "region", "city",
            "revenue_status", "employee_count", "female_founder"]
    st.dataframe(df[show].sort_values("funding_date", ascending=False))
    st.download_button("Download filtered data (CSV)", df[show].to_csv(index=False).encode("utf-8"),
                       file_name="filtered_funding_rounds.csv", mime="text/csv")
