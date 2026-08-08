import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


DATA_FILE = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "scifact_dashboard_data.json"
)


st.set_page_config(
    page_title="SciFact Retrieval Analytics",
    page_icon="🔎",
    layout="wide",
)


@st.cache_data
def load_data():
    with DATA_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def format_score(value, decimals=4):
    if value is None:
        return "—"

    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return "—"


try:
    data = load_data()
except Exception as exc:
    st.error(
        f"Could not load SciFact dashboard data: {exc}"
    )
    st.stop()


metadata = data.get("metadata", {})
metrics = data.get("metrics", [])
significance = data.get("significance", [])
cases = data.get("cases", [])

if not cases:
    st.error("No SciFact claims were found.")
    st.stop()


st.title("SciFact Evidence-Retrieval Analytics")

st.caption(
    "Traceable comparison of sparse retrieval, dense retrieval, "
    "reciprocal-rank fusion, and cross-encoder reranking."
)


with st.sidebar:
    st.header("SciFact controls")

    claim_labels = {
        case["claim_id"]: (
            f'{case["claim_id"]} — '
            f'{case.get("claim", "")[:75]}'
        )
        for case in cases
    }

    selected_claim_id = st.selectbox(
        "Select scientific claim",
        options=list(claim_labels.keys()),
        format_func=lambda claim_id: (
            claim_labels[claim_id]
        ),
    )

    top_k = st.selectbox(
        "Documents to display",
        options=[5, 10, 20],
        index=1,
    )

    st.divider()

    st.caption(
        f'Dataset: {metadata.get("dataset", "SciFact")}\n\n'
        f'Ranked development claims: '
        f'{metadata.get("ranked_claims", len(cases))}\n\n'
        f'Claims used for evaluation: '
        f'{metadata.get("evaluated_claims", "—")}\n\n'
        f'Corpus documents: '
        f'{metadata.get("corpus_documents", "—")}'
    )


selected_case = next(
    case
    for case in cases
    if case["claim_id"] == selected_claim_id
)


overview_tab, explorer_tab, provenance_tab = st.tabs(
    [
        "📊 Retrieval overview",
        "🔎 Claim explorer",
        "📄 Document provenance",
    ]
)


with overview_tab:
    st.subheader("Global retrieval performance")

    st.info(
        "The claim explorer contains rankings for all 300 SciFact "
        "development claims. Retrieval metrics and paired-bootstrap "
        "comparisons are computed on the 188 claims with released "
        "gold evidence annotations."
    )

    metrics_df = pd.DataFrame(metrics)

    metric_columns = [
        column
        for column in [
            "system",
            "MRR",
            "Recall@10",
            "nDCG@10",
            "MAP@1000",
        ]
        if column in metrics_df.columns
    ]

    st.dataframe(
        metrics_df[metric_columns],
        hide_index=True,
        use_container_width=True,
    )

    available_metrics = [
        metric
        for metric in [
            "MRR",
            "Recall@10",
            "nDCG@10",
            "MAP@1000",
        ]
        if metric in metrics_df.columns
        and metrics_df[metric].notna().any()
    ]

    if available_metrics:
        selected_metric = st.selectbox(
            "Metric for visual comparison",
            options=available_metrics,
            index=0,
        )

        chart_data = metrics_df[
            ["system", selected_metric]
        ].dropna()

        chart = (
            alt.Chart(chart_data)
            .mark_bar()
            .encode(
                x=alt.X(
                    "system:N",
                    title=None,
                    sort=[
                        "BM25",
                        "E5",
                        "RRF",
                        "Cross-encoder",
                    ],
                    axis=alt.Axis(labelAngle=0),
                ),
                y=alt.Y(
                    f"{selected_metric}:Q",
                    title=selected_metric,
                    scale=alt.Scale(
                        domain=[0, 1]
                    ),
                ),
                tooltip=[
                    alt.Tooltip(
                        "system:N",
                        title="System",
                    ),
                    alt.Tooltip(
                        f"{selected_metric}:Q",
                        title=selected_metric,
                        format=".4f",
                    ),
                ],
            )
        )

        labels = (
            alt.Chart(chart_data)
            .mark_text(
                dy=-8,
                fontSize=14,
            )
            .encode(
                x=alt.X(
                    "system:N",
                    sort=[
                        "BM25",
                        "E5",
                        "RRF",
                        "Cross-encoder",
                    ],
                ),
                y=alt.Y(
                    f"{selected_metric}:Q"
                ),
                text=alt.Text(
                    f"{selected_metric}:Q",
                    format=".4f",
                ),
            )
        )

        st.altair_chart(
            chart + labels,
            width="stretch",
        )

    if significance:
        st.divider()
        st.subheader("Paired-bootstrap comparisons")

        significance_df = pd.DataFrame(
            significance
        )

        st.dataframe(
            significance_df,
            hide_index=True,
            use_container_width=True,
        )

        st.caption(
            "Confidence intervals and adjusted p-values are shown "
            "as recorded by the frozen significance-analysis output."
        )


with explorer_tab:
    st.subheader("Scientific claim")

    st.info(
        selected_case.get(
            "claim",
            "",
        )
    )

    documents = selected_case.get(
        "top_documents",
        [],
    )[:top_k]

    top_document = (
        documents[0]
        if documents
        else {}
    )

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric(
        "Claim ID",
        selected_case.get(
            "claim_id",
            "—",
        ),
    )

    metric2.metric(
        "Reranked candidates",
        selected_case.get(
            "candidate_count",
            "—",
        ),
    )

    metric3.metric(
        "Top document",
        top_document.get(
            "document_id",
            "—",
        ),
    )

    metric4.metric(
        "Top cross-encoder score",
        format_score(
            top_document.get(
                "cross_encoder_score"
            ),
            4,
        ),
    )

    table_rows = []

    for document in documents:
        table_rows.append(
            {
                "CE rank": document.get(
                    "cross_encoder_rank"
                ),
                "Document ID": document.get(
                    "document_id"
                ),
                "Title": document.get(
                    "title"
                ),
                "CE score": document.get(
                    "cross_encoder_score"
                ),
                "BM25 rank": document.get(
                    "bm25_rank"
                ),
                "BM25 score": document.get(
                    "bm25_score"
                ),
                "E5 rank": document.get(
                    "e5_rank"
                ),
                "E5 score": document.get(
                    "e5_score"
                ),
                "RRF rank": document.get(
                    "rrf_rank"
                ),
                "RRF score": document.get(
                    "rrf_score"
                ),
            }
        )

    results_df = pd.DataFrame(table_rows)

    st.dataframe(
        results_df,
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "Cross-encoder scores, BM25 scores, dense cosine scores, "
        "and RRF scores have different scales and should not be "
        "compared directly as numerical magnitudes."
    )


with provenance_tab:
    st.subheader("Retrieved-document provenance")

    documents = selected_case.get(
        "top_documents",
        [],
    )[:top_k]

    for document in documents:
        document_id = document.get(
            "document_id",
            "unknown",
        )

        title = document.get(
            "title",
            f"SciFact document {document_id}",
        )

        label = (
            f'Rank {document.get("cross_encoder_rank", "—")} — '
            f'{document_id} — {title}'
        )

        with st.expander(
            label,
            expanded=(
                document.get(
                    "cross_encoder_rank"
                ) == 1
            ),
        ):
            score1, score2, score3, score4 = st.columns(4)

            score1.metric(
                "Cross-encoder",
                format_score(
                    document.get(
                        "cross_encoder_score"
                    ),
                    4,
                ),
            )

            score2.metric(
                "BM25 rank",
                document.get(
                    "bm25_rank",
                    "—",
                ),
            )

            score3.metric(
                "E5 rank",
                document.get(
                    "e5_rank",
                    "—",
                ),
            )

            score4.metric(
                "RRF rank",
                document.get(
                    "rrf_rank",
                    "—",
                ),
            )

            st.markdown(f"### {title}")

            abstract = document.get(
                "abstract",
                "",
            )

            if abstract:
                st.text_area(
                    "Scientific abstract",
                    value=abstract,
                    height=300,
                    disabled=True,
                    key=(
                        f'{selected_claim_id}_'
                        f'{document_id}'
                    ),
                )
            else:
                st.info(
                    "No abstract text was found "
                    "for this document."
                )

    st.caption(
        "Each displayed document retains its SciFact identifier, "
        "title, source abstract, and rankings from all four "
        "retrieval configurations."
    )
