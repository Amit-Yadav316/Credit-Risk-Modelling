"""Underwriter-facing scoring console."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from credit_risk.config import AppConfig
from credit_risk.scoring.service import ScoringService

st.set_page_config(page_title="Credit Risk Scorer", page_icon=":bar_chart:", layout="wide")

# Same domains the API validates against, and the same delinquency sentinel the
# adapter writes: blank in Lending Club means never delinquent, which is the best
# outcome rather than a missing value.
NEVER_DELINQUENT = 999.0
NO_PUBLIC_RECORD = 999.0
PURPOSES = [
    "debt_consolidation", "credit_card", "home_improvement", "other", "major_purchase",
    "small_business", "car", "medical", "moving", "vacation", "house", "wedding",
    "renewable_energy", "educational",
]
EMPLOYMENT_TYPES = ["long_tenure", "mid", "junior", "unknown"]
RESIDENCE_TYPES = ["owned", "rented", "family"]
VERIFICATION_TYPES = ["source_verified", "verified", "not_verified"]
STATES = [
    "AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "HI", "IA", "ID",
    "IL", "IN", "KS", "KY", "LA", "MA", "MD", "ME", "MI", "MN", "MO", "MS", "MT", "NC",
    "ND", "NE", "NH", "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA", "RI", "SC", "SD",
    "TN", "TX", "UT", "VA", "VT", "WA", "WI", "WV", "WY",
]


@st.cache_resource
def load_service(alias: str) -> ScoringService:
    return ScoringService(AppConfig.load(), alias=alias)


def gauge(score: int, cfg) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"font": {"size": 52}},
            gauge={
                "axis": {"range": [cfg.min_score, cfg.max_score]},
                "bar": {"color": "#1f2937", "thickness": 0.25},
                "steps": [
                    {"range": [cfg.min_score, cfg.policy["refer_above"]], "color": "#fca5a5"},
                    {
                        "range": [cfg.policy["refer_above"], cfg.policy["approve_above"]],
                        "color": "#fcd34d",
                    },
                    {"range": [cfg.policy["approve_above"], cfg.max_score], "color": "#86efac"},
                ],
            },
        )
    )
    fig.update_layout(height=280, margin=dict(t=20, b=10, l=20, r=20))
    return fig


config = AppConfig.load()
st.title("Credit risk scoring console")

with st.sidebar:
    st.subheader("Model")
    alias = st.radio("Registry alias", ["champion", "challenger"], horizontal=True)
    service = load_service(alias)
    st.caption(f"{config.mlflow.registered_model} @ {alias}")
    st.divider()
    st.subheader("Policy cutoffs")
    st.write(f"Approve at or above {config.scorecard.policy['approve_above']}")
    refer_lo = config.scorecard.policy["refer_above"]
    refer_hi = config.scorecard.policy["approve_above"] - 1
    st.write(f"Refer between {refer_lo} and {refer_hi}")

tab_single, tab_batch = st.tabs(["Single application", "Batch file"])

with tab_single:
    c1, c2, c3 = st.columns(3)
    with c1:
        loan_amount = st.number_input("Loan amount ($)", 500, 40_000, 12_000, step=500)
        annual_income = st.number_input("Annual income ($)", 1_200, 250_000, 65_000, step=1_000)
        tenure_months = st.selectbox("Term (months)", [36, 60], index=0)
        interest_rate = st.slider("Interest rate (%)", 5.0, 29.0, 12.4, 0.01)
        debt_to_income = st.slider("Debt to income (%)", 0.0, 60.0, 17.5, 0.1)
        installment_amount = st.number_input("Monthly payment ($)", 10.0, 10_000.0, 401.2, step=10.0)
    with c2:
        bureau_score = st.slider("FICO score", 600, 850, 697)
        credit_utilisation = st.slider("Revolving utilisation", 0.0, 2.0, 0.54, 0.01)
        never_delinquent = st.checkbox("Never delinquent", value=True)
        months_since_delinq = (
            NEVER_DELINQUENT
            if never_delinquent
            else float(st.slider("Months since last delinquency", 0, 150, 18))
        )
        enquiries_6m = st.number_input("Enquiries (6m)", 0, 50, 0)
        no_public_record = st.checkbox("No public record", value=True)
        months_since_public_record = (
            NO_PUBLIC_RECORD
            if no_public_record
            else float(st.slider("Months since public record", 0, 120, 66))
        )
    with c3:
        num_open_accounts = st.number_input("Open accounts", 0, 150, 11)
        num_delinquent_accounts = st.number_input("Delinquencies (2y)", 0, 50, 0)
        public_records = st.number_input("Public records", 0, 100, 0)
        total_accounts = st.number_input("Accounts ever opened", 1, 200, 24)
        revolving_balance = st.number_input("Revolving balance ($)", 0, 1_000_000, 12_000, step=500)
        credit_history_months = st.slider("Credit history (months)", 6, 720, 190)
        income_verification = st.selectbox("Income verification", VERIFICATION_TYPES)
        loan_purpose = st.selectbox("Purpose", PURPOSES)
        employment_type = st.selectbox("Employment length", EMPLOYMENT_TYPES)
        residence_type = st.selectbox("Home ownership", RESIDENCE_TYPES)
        state = st.selectbox("State", STATES, index=STATES.index("CA"))

    if st.button("Score application", type="primary", use_container_width=True):
        payload = dict(
            loan_amount=loan_amount,
            annual_income=annual_income,
            tenure_months=tenure_months,
            interest_rate=interest_rate,
            num_open_accounts=num_open_accounts,
            num_delinquent_accounts=num_delinquent_accounts,
            credit_utilisation=credit_utilisation,
            bureau_score=bureau_score,
            debt_to_income=debt_to_income,
            months_since_delinq=months_since_delinq,
            months_since_public_record=months_since_public_record,
            public_records=public_records,
            installment_amount=installment_amount,
            revolving_balance=revolving_balance,
            total_accounts=total_accounts,
            credit_history_months=credit_history_months,
            income_verification=income_verification,
            enquiries_6m=enquiries_6m,
            loan_purpose=loan_purpose,
            employment_type=employment_type,
            residence_type=residence_type,
            state=state,
        )
        try:
            result = service.score_one(payload)
        except Exception as exc:
            st.error(f"Could not reach the model registry: {exc}")
        else:
            left, right = st.columns([1, 1])
            with left:
                st.plotly_chart(gauge(result.score, config.scorecard), use_container_width=True)
            with right:
                st.metric("Decision", result.decision)
                st.metric("Probability of default", f"{result.probability_of_default:.2%}")
                st.metric("Risk band", result.band)
                st.caption(f"Model version {result.model_version} | LTI {loan_amount / annual_income:.2f}")
                if result.reasons:
                    st.write("Top risk drivers")
                    for r in result.reasons:
                        st.write(f"- {r}")

with tab_batch:
    upload = st.file_uploader("Upload a CSV of applications", type="csv")
    if upload is not None:
        frame = pd.read_csv(upload)
        scored = service.score_batch(frame)
        out = pd.concat([frame.reset_index(drop=True), scored], axis=1)
        st.dataframe(out.head(200), use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Applications", f"{len(out):,}")
        c2.metric("Approve rate", f"{(out.decision == 'Approve').mean():.1%}")
        c3.metric("Median score", int(out.score.median()))
        st.download_button(
            "Download scored file",
            out.to_csv(index=False).encode(),
            "scored_applications.csv",
            use_container_width=True,
        )
