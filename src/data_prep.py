"""Load, clean, merge and validate the four raw CSV files.

Run from the project root:  python src/data_prep.py
Outputs:
    data/processed/funding_master.csv   one row per funding round + startup attributes
    reports/data_quality.json           every check below, with its result
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import MASTER_CSV, RAW_DIR, REPORT_DIR, STAGE_ORDER  # noqa: E402


def load_raw():
    rounds = pd.read_csv(RAW_DIR / "funding_rounds.csv", parse_dates=["funding_date"])
    startups = pd.read_csv(RAW_DIR / "startups.csv")
    investors = pd.read_csv(RAW_DIR / "investors.csv")
    quarterly = pd.read_csv(RAW_DIR / "quarterly_summary.csv")
    return rounds, startups, investors, quarterly


def build_master(rounds=None, startups=None):
    """One row per funding round, joined to startup-level attributes."""
    if rounds is None or startups is None:
        rounds, startups, _, _ = load_raw()

    keep = ["startup_id", "founded_year", "employee_count", "female_founder", "revenue_status"]
    df = rounds.merge(startups[keep], on="startup_id", how="left", validate="many_to_one")

    df["funding_stage"] = pd.Categorical(df["funding_stage"], STAGE_ORDER, ordered=True)
    df["period"] = df["year"].astype(str) + "-" + df["quarter"]
    df["log_amount"] = np.log10(df["amount_million_usd"])

    # Startup age at the time of the round. The raw data contains rounds dated
    # before the startup's founded_year, which gives a negative age. Keep the
    # rows, flag them, and clip the age at 0 for modelling.
    df["age_at_round_raw"] = df["year"] - df["founded_year"]
    df["funded_before_founding"] = (df["age_at_round_raw"] < 0).astype(int)
    df["age_at_round"] = df["age_at_round_raw"].clip(lower=0)

    # Chronological round number within each startup.
    df = df.sort_values(["startup_id", "funding_date", "round_id"]).reset_index(drop=True)
    df["round_number"] = df.groupby("startup_id").cumcount() + 1
    df["female_founder"] = df["female_founder"].astype(int)
    return df


def run_quality_checks(df, rounds, startups, investors, quarterly):
    """Return a dict of facts about the data. Nothing here modifies the data."""
    q = {}
    q["rows_funding_rounds"] = int(len(rounds))
    q["rows_startups"] = int(len(startups))
    q["rows_investors"] = int(len(investors))
    q["rows_quarterly_summary"] = int(len(quarterly))
    q["unique_startups_in_rounds"] = int(rounds["startup_id"].nunique())
    q["unique_countries"] = int(rounds["country"].nunique())
    q["unique_sectors"] = int(rounds["sector"].nunique())
    q["date_min"] = str(rounds["funding_date"].min().date())
    q["date_max"] = str(rounds["funding_date"].max().date())
    q["missing_values_total"] = int(rounds.isna().sum().sum())
    q["duplicate_rows"] = int(rounds.duplicated().sum())
    q["round_id_unique"] = bool(rounds["round_id"].is_unique)

    # Do startup-level attributes in the rounds table match the startups table?
    m = rounds.merge(startups, on="startup_id", suffixes=("", "_s"))
    q["rounds_attrs_match_startups_table"] = bool(
        all((m[c] == m[c + "_s"]).all() for c in ["sector", "country", "region", "city"])
    )

    # Arithmetic checks on the money columns.
    q["post_equals_pre_plus_amount_pct"] = float(np.isclose(
        rounds["pre_money_valuation_million_usd"] + rounds["amount_million_usd"],
        rounds["post_money_valuation_million_usd"], atol=0.02).mean() * 100)
    implied = rounds["amount_million_usd"] / rounds["post_money_valuation_million_usd"] * 100
    q["dilution_matches_amount_over_post_pct"] = float(
        ((implied - rounds["equity_dilution_pct"]).abs() < 0.5).mean() * 100)

    # Date / year / quarter consistency.
    q["year_matches_date_pct"] = float((rounds["funding_date"].dt.year == rounds["year"]).mean() * 100)
    q["quarter_matches_date_pct"] = float(
        (("Q" + rounds["funding_date"].dt.quarter.astype(str)) == rounds["quarter"]).mean() * 100)

    # Logical problems.
    q["rounds_dated_before_founding_year"] = int(df["funded_before_founding"].sum())
    order = {s: i for i, s in enumerate(STAGE_ORDER)}
    tmp = df.assign(o=df["funding_stage"].map(order).astype(int)).sort_values(["startup_id", "funding_date"])
    q["startups_with_out_of_order_stages"] = int(
        tmp.groupby("startup_id")["o"].apply(lambda s: (s.diff().dropna() < 0).any()).sum())
    q["partial_year"] = {"year": int(rounds["year"].max()),
                         "quarters_present": sorted(rounds.loc[rounds["year"] == rounds["year"].max(), "quarter"].unique().tolist())}

    # Investor table vs rounds table.
    counts = rounds["lead_investor"].value_counts()
    inv = investors.set_index("investor_name")
    q["lead_investors_all_in_investor_table"] = bool(rounds["lead_investor"].isin(inv.index).all())
    q["investor_deal_counts_match_rounds"] = bool(
        (inv["total_deals_in_dataset"] == counts.reindex(inv.index).fillna(0).astype(int)).all())
    q["investor_type_in_rounds_equals_investor_table_pct"] = float(
        (rounds["investor_type"] == rounds["lead_investor"].map(inv["investor_type"])).mean() * 100)

    # Quarterly summary vs rebuilt from rounds.
    rebuilt = (rounds.groupby(["year", "quarter"])
               .agg(total_deals=("round_id", "count"), total_funding_million_usd=("amount_million_usd", "sum"))
               .reset_index().merge(quarterly, on=["year", "quarter"], suffixes=("", "_given")))
    q["quarterly_summary_matches_rounds"] = bool(
        (rebuilt["total_deals"] == rebuilt["total_deals_given"]).all()
        and np.allclose(rebuilt["total_funding_million_usd"], rebuilt["total_funding_million_usd_given"], atol=0.05))
    return q


def load_master():
    """Load the processed master table (used by the EDA, model and dashboard)."""
    if not MASTER_CSV.exists():
        main()
    df = pd.read_csv(MASTER_CSV, parse_dates=["funding_date"])
    df["funding_stage"] = pd.Categorical(df["funding_stage"], STAGE_ORDER, ordered=True)
    return df


def main():
    rounds, startups, investors, quarterly = load_raw()
    df = build_master(rounds, startups)
    MASTER_CSV.parent.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(MASTER_CSV, index=False)
    checks = run_quality_checks(df, rounds, startups, investors, quarterly)
    (REPORT_DIR / "data_quality.json").write_text(json.dumps(checks, indent=2))
    print(f"Master table: {df.shape[0]} rows x {df.shape[1]} columns -> {MASTER_CSV}")
    print(json.dumps(checks, indent=2))
    return df


if __name__ == "__main__":
    main()
