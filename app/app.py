"""Streamlit demo: whom should the SMS budget reach?

Reads only the aggregated outputs of notebook 06 (app/data/*.csv): no model is trained here and
no client level data is used. Run from the repository root:  streamlit run app/app.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

DATA = Path(__file__).parent / "data"
BLUE, YELLOW, INK, INK_2, AXIS, SURFACE = "#2a78d6", "#eda100", "#0b0b0b", "#52514e", "#c3c2b7", "#fcfcfb"
POLICIES = {                         # display name: column prefix in policy_curves.csv
    "Uplift ranking": "uplift",
    "Likely buyers first": "purchase_probability",
    "Random": "random",
}
PROFILE_ROWS = {"predicted uplift (points)": "Predicted uplift (points)",
                "purchase probability without SMS (%)": "Buys without SMS (%)",
                "median age": "Median age",
                "median visits, last 30 days": "Visits, last 30 days",
                "median days since card issue": "Days since card issue"}


@st.cache_data
def load():
    return pd.read_csv(DATA / "policy_curves.csv"), pd.read_csv(DATA / "audience.csv")


def at_budget(curves, budget_pct):
    """Row of the curves at a budget given in percent (the curves have one row per percent)."""
    return curves.iloc[int(round(budget_pct))]


def outcome(row, prefix, share, customers, cost, margin):
    """Extra purchases, messages and profit (with 95% interval) when `share` of customers is contacted."""
    extra = {k: customers * row[f"{prefix}{s}"] for k, s in [("est", ""), ("low", "_low"), ("high", "_high")]}
    messages = customers * share
    profit = {k: v * margin - messages * cost for k, v in extra.items()}
    return extra, messages, profit


def suggested_budget(curves, cost_ratio):
    """Budget with the highest profit for the uplift ranking on the training folds (as in notebook 06)."""
    profit = curves["uplift_train"] - cost_ratio * curves["share_contacted"]
    i = int(profit.idxmax())
    return curves.loc[i, "share_contacted"], profit.iloc[i] > 0


def fmt(x, digits=0):
    return f"{x:,.{digits}f}"


st.set_page_config(page_title="Whom should the SMS reach?", layout="wide")
curves, audience = load()

st.title("Whom should the SMS budget reach?")
st.markdown(
    "A retailer can send an SMS to part of its customers. Sent to the customers most likely to buy anyway, "
    "it changes little; sent to the customers it actually moves, each message creates more extra purchases. "
    "Results come from a randomized SMS test of the X5 retail chain (2019), measured on 60,012 customers that "
    "no model saw."
)

with st.sidebar:
    st.header("Your campaign")
    budget = st.slider("Share of customers contacted (%)", 1, 100, 10,
                       help="How many of your customers the budget allows you to text.")
    customers = st.number_input("Customers in your base", min_value=1_000, max_value=10_000_000, value=100_000,
                                step=10_000, help="Used only to scale the results to your size.")
    cost = st.number_input("Cost of one SMS", min_value=0.0, value=0.05, step=0.01, format="%.2f",
                           help="In any currency unit; only the ratio to the margin matters.")
    margin = st.number_input("Margin of one extra purchase", min_value=0.01, value=1.00, step=0.10, format="%.2f",
                             help="What one extra purchasing customer earns you, in the same unit as the cost.")
    st.caption("Cost and margin are assumptions: the data contains neither prices nor margins.")

share = budget / 100
ratio = cost / margin
row = at_budget(curves, budget)

# Policy results at the chosen budget
st.subheader(f"Contacting {budget}% of {fmt(customers)} customers ({fmt(customers * share)} messages)")
cols = st.columns(4)
results = {name: outcome(row, prefix, share, customers, cost, margin) for name, prefix in POLICIES.items()}
results["Everyone"] = outcome(curves.iloc[-1], "random", 1.0, customers, cost, margin)
for col, (name, (extra, messages, profit)) in zip(cols, results.items()):
    with col:
        st.markdown(f"**{name if name != 'Everyone' else 'Everyone (100%)'}**")
        st.metric("Extra purchases", fmt(extra["est"]), help=f"95% interval: {fmt(extra['low'])} to {fmt(extra['high'])}")
        st.metric("Profit", fmt(profit["est"]), help=f"95% interval: {fmt(profit['low'])} to {fmt(profit['high'])}")
        st.caption(f"{fmt(messages)} messages; profit interval {fmt(profit['low'])} to {fmt(profit['high'])}")

best_share, pays = suggested_budget(curves, ratio)
if pays:
    st.info(f"At these costs, the most profitable budget for the uplift ranking is about **{best_share:.0%}** of "
            f"customers (chosen on the training data). Contacting a customer pays only if the SMS raises their "
            f"purchase probability by more than {ratio:.1%}.")
else:
    st.warning(f"At these costs no budget pays: the SMS would have to raise purchase probability by more than "
               f"{ratio:.1%} for the customers contacted.")

# Profit curve
left, right = st.columns([3, 2])
with left:
    pct = 100 * curves["share_contacted"]
    fig, ax = plt.subplots(figsize=(7, 3.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for name, prefix, color, style in [("Uplift ranking", "uplift", BLUE, "-"),
                                       ("Likely buyers first", "purchase_probability", YELLOW, "-"),
                                       ("Random", "random", INK_2, "--")]:
        prof = lambda col: customers * (curves[col] * margin - curves["share_contacted"] * cost)
        if prefix == "uplift":
            ax.fill_between(pct, prof(f"{prefix}_low"), prof(f"{prefix}_high"), color=color, alpha=0.2, linewidth=0)
        ax.plot(pct, prof(prefix), color=color, linestyle=style, linewidth=2.2 if style == "-" else 1.4, label=name)
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.axvline(budget, color=INK, linewidth=1, linestyle=":")
    ax.set_xlabel("Share of customers contacted (%)")
    ax.set_ylabel("Profit")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlim(0, 100)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#e1e0d9")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Profit by budget (band: 95% interval for the uplift ranking)", loc="left", fontsize=10)
    st.pyplot(fig)

# Who is contacted
with right:
    step = min(1.0, max(0.05, np.ceil(share * 20) / 20))          # audience is stored in 5% steps
    a = audience.set_index("budget")
    targeted, everyone = a.loc[round(step, 2)], a.loc[1.0]
    st.markdown(f"**Who the uplift ranking contacts** (top {step:.0%})")
    rows_ = list(PROFILE_ROWS)
    profile_table = pd.DataFrame({"Contacted": targeted[rows_], "All": everyone[rows_]}).rename(index=PROFILE_ROWS)
    st.dataframe(profile_table.round(1), use_container_width=True)
    segments = [c for c in audience.columns if c not in ["budget", "clients", "redeemed points before (%)", *PROFILE_ROWS]]
    seg = pd.DataFrame({"Contacted": targeted[segments], "All": everyone[segments]}).mul(100).round(1)
    st.dataframe(seg.sort_values("Contacted", ascending=False).rename_axis("Segment (%)"),
                 use_container_width=True)

with st.expander("Assumptions and limits"):
    st.markdown(
        "* One randomized SMS campaign of one Russian grocery chain in 2019, among recently active customers. "
        "Results may not carry over to other retailers, countries or messages.\n"
        "* An extra purchase means one more customer who bought during the campaign window. Whether the SMS "
        "created new purchases or moved planned ones earlier, and how much was spent, is not in the data.\n"
        "* Cost and margin are your assumptions. Intervals are 95% bootstrap intervals from the test set and "
        "do not include uncertainty about cost or margin.\n"
        "* The ranking was learned on one campaign; a follow-up randomized test should confirm it before use.\n"
        "* Aggregated results only: no customer level data is used in this app."
    )
