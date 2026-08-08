import json
from pathlib import Path

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent

PLABA_FILE = (
    BASE_DIR
    / "data"
    / "plaba_dashboard_data.json"
)

SCIFACT_FILE = (
    BASE_DIR
    / "data"
    / "scifact_dashboard_data.json"
)


st.set_page_config(
    page_title="Evidence-to-Insight BI Dashboard",
    page_icon="📊",
    layout="wide",
)


def load_json(path):
    if not path.exists():
        return {}

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


plaba = load_json(PLABA_FILE)
scifact = load_json(SCIFACT_FILE)

plaba_cases = plaba.get("cases", [])
scifact_cases = scifact.get("cases", [])


st.title("Evidence-to-Insight Business Intelligence Dashboard")

st.caption(
    "An integrated prototype for scientific evidence retrieval, "
    "plain-language adaptation, consistency screening, provenance, "
    "and conservative source fallback."
)


scifact_metadata = scifact.get("metadata", {})

metric1, metric2, metric3, metric4, metric5 = st.columns(5)

metric1.metric(
    "SciFact ranked claims",
    scifact_metadata.get(
        "ranked_claims",
        len(scifact_cases),
    ),
)

metric2.metric(
    "SciFact evaluated claims",
    scifact_metadata.get(
        "evaluated_claims",
        "—",
    ),
)

metric3.metric(
    "PLABA test cases",
    len(plaba_cases),
)

metric4.metric(
    "PLABA candidates",
    sum(
        len(case.get("candidates", []))
        for case in plaba_cases
    ),
)

fallback_count = sum(
    bool(
        case.get(
            "guard_is_source_fallback",
            False,
        )
    )
    for case in plaba_cases
)

metric5.metric(
    "PLABA source fallbacks",
    fallback_count,
)


st.divider()

left, right = st.columns(2)

with left:
    st.subheader("🔎 SciFact Retrieval Analytics")

    st.markdown(
        """
Explore scientific claims and compare:

- BM25 sparse retrieval
- E5 dense retrieval
- reciprocal-rank fusion
- cross-encoder reranking
- top retrieved documents and provenance
- global retrieval metrics and significance results
"""
    )

with right:
    st.subheader("✍️ PLABA Adaptation Analytics")

    st.markdown(
        """
Explore biomedical source texts and compare:

- Beam-4, Greedy and Concise candidates
- readability and compression indicators
- number, negation and uncertainty checks
- frozen selector and guard decisions
- source fallback and test-set KPIs
"""
    )


st.divider()

st.info(
    "Use the page navigation in the sidebar to open "
    "SciFact Retrieval Analytics or PLABA Adaptation Analytics."
)

st.caption(
    "The prototype demonstrates component integration and traceability. "
    "It has not been evaluated through a user study, and no claim is made "
    "that it improves real-world decision quality."
)
