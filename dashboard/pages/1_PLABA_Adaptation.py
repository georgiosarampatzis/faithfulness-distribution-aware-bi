import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "plaba_dashboard_data.json"

st.set_page_config(
    page_title="Faithfulness-Aware BI Dashboard",
    page_icon="📊",
    layout="wide",
)


@st.cache_data
def load_data():
    with DATA_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def pass_fail(value):
    return "✅ Pass" if value else "❌ Fail"


def display_candidate_name(candidate_type):
    names = {
        "beam4": "Beam-4",
        "greedy": "Greedy",
        "concise": "Concise",
        "source_fallback": "Source fallback",
    }

    return names.get(
        str(candidate_type).lower(),
        str(candidate_type),
    )


def format_number(value, decimals=2):
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return "—"


def format_integer(value):
    try:
        return str(int(round(float(value))))
    except (TypeError, ValueError):
        return "—"


try:
    data = load_data()
except Exception as exc:
    st.error(f"Could not load dashboard data: {exc}")
    st.stop()


cases = data.get("cases", [])
metadata = data.get("metadata", {})

if not cases:
    st.error("No dashboard cases found.")
    st.stop()


st.title("Faithfulness-Aware Plain-Language Dashboard")

st.caption(
    "Traceable biomedical plain-language adaptation, "
    "reference-free candidate selection, consistency screening, "
    "and conservative source fallback."
)


with st.sidebar:
    st.header("Dashboard controls")

    case_labels = {
        case["case_id"]: (
            f'{case["case_id"]} — '
            f'{case.get("short_title", "Biomedical case")}'
        )
        for case in cases
    }

    selected_id = st.selectbox(
        "Select PLABA test case",
        options=list(case_labels.keys()),
        format_func=lambda case_id: case_labels[case_id],
    )

    mode = st.radio(
        "Operating mode",
        options=[
            "Quality-first",
            "Reliability-first",
        ],
        index=1,
    )

    st.divider()

    st.markdown(
        """
**Quality-first**

Returns the fixed Beam-4 adaptation.

**Reliability-first**

Reproduces the frozen full-consistency guard decision.  
The dashboard does not recalculate the decision.
"""
    )

    st.divider()

    st.caption(
        f'Dataset: {metadata.get("dataset", "PLABA")}\n\n'
        f'Split: {metadata.get("split", "untouched test")}\n\n'
        f'Cases: {metadata.get("examples", len(cases))}'
    )


case = next(
    item
    for item in cases
    if item["case_id"] == selected_id
)

source = case.get("source", {})
candidates = case.get("candidates", [])

eligible_candidates = [
    candidate
    for candidate in candidates
    if candidate.get("eligible", False)
]

guard_decision = case.get(
    "guard_decision",
    "unknown",
)

guard_is_fallback = bool(
    case.get(
        "guard_is_source_fallback",
        False,
    )
)


st.subheader("Case context")

context_col1, context_col2 = st.columns([1, 3])

context_col1.metric(
    "PMID",
    case.get("pmid", source.get("document_id", "—")),
)

context_col2.info(
    case.get(
        "question",
        "No consumer-question information is available.",
    )
)


metric1, metric2, metric3, metric4 = st.columns(4)

metric1.metric(
    "Source words",
    format_integer(source.get("word_count")),
)

metric2.metric(
    "Generated candidates",
    len(candidates),
)

metric3.metric(
    "Eligible candidates",
    len(eligible_candidates),
)

metric4.metric(
    "Guard decision",
    display_candidate_name(guard_decision),
)


overview_tab, source_tab, candidates_tab, decision_tab = st.tabs(
    [
        "📊 Test-set overview",
        "📄 Original source",
        "✍️ Adaptation candidates",
        "🛡️ Final decision",
    ]
)



with overview_tab:
    st.subheader("Untouched PLABA test-set overview")

    all_candidates = [
        candidate
        for test_case in cases
        for candidate in test_case.get("candidates", [])
    ]

    total_cases = len(cases)
    total_candidates = len(all_candidates)

    eligible_outputs = sum(
        bool(candidate.get("eligible", False))
        for candidate in all_candidates
    )

    fallback_cases = sum(
        bool(test_case.get("guard_is_source_fallback", False))
        for test_case in cases
    )

    fallback_rate = (
        100.0 * fallback_cases / total_cases
        if total_cases
        else 0.0
    )

    overview_metric1, overview_metric2, overview_metric3, \
        overview_metric4, overview_metric5 = st.columns(5)

    overview_metric1.metric(
        "Test cases",
        total_cases,
    )

    overview_metric2.metric(
        "Generated candidates",
        total_candidates,
    )

    overview_metric3.metric(
        "Eligible candidate outputs",
        eligible_outputs,
    )

    overview_metric4.metric(
        "Source fallbacks",
        fallback_cases,
    )

    overview_metric5.metric(
        "Fallback rate",
        f"{fallback_rate:.2f}%",
    )

    st.caption(
        "The overview is computed directly from the frozen selector "
        "features and locked full-consistency guard decisions."
    )

    st.divider()

    guard_labels = [
        display_candidate_name(
            test_case.get("guard_decision", "unknown")
        )
        for test_case in cases
    ]

    guard_order = [
        "Beam-4",
        "Greedy",
        "Concise",
        "Source fallback",
    ]

    guard_counts = {
        label: guard_labels.count(label)
        for label in guard_order
    }

    guard_df = pd.DataFrame(
        {
            "Guard decision": list(guard_counts.keys()),
            "Cases": list(guard_counts.values()),
        }
    )

    candidate_order = [
        ("beam4", "Beam-4"),
        ("greedy", "Greedy"),
        ("concise", "Concise"),
    ]

    eligibility_rows = []

    for candidate_type, display_name in candidate_order:
        candidate_subset = [
            candidate
            for candidate in all_candidates
            if candidate.get("candidate_type") == candidate_type
        ]

        eligible_count = sum(
            bool(candidate.get("eligible", False))
            for candidate in candidate_subset
        )

        total_count = len(candidate_subset)

        eligibility_rows.append(
            {
                "Candidate": display_name,
                "Eligible": eligible_count,
                "Rejected": total_count - eligible_count,
                "Total": total_count,
                "Eligibility rate": (
                    100.0 * eligible_count / total_count
                    if total_count
                    else 0.0
                ),
            }
        )

    eligibility_df = pd.DataFrame(eligibility_rows)

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("#### Guard-decision distribution")

        guard_maximum = int(guard_df["Cases"].max()) + 12

        guard_bars = (
            alt.Chart(guard_df)
            .mark_bar()
            .encode(
                y=alt.Y(
                    "Guard decision:N",
                    sort=guard_order,
                    title=None,
                    axis=alt.Axis(labelLimit=160),
                ),
                x=alt.X(
                    "Cases:Q",
                    title="Test cases",
                    scale=alt.Scale(
                        domain=[0, guard_maximum]
                    ),
                ),
                tooltip=[
                    alt.Tooltip(
                        "Guard decision:N",
                        title="Decision",
                    ),
                    alt.Tooltip(
                        "Cases:Q",
                        title="Cases",
                    ),
                ],
            )
        )

        guard_labels_chart = (
            alt.Chart(guard_df)
            .mark_text(
                align="left",
                baseline="middle",
                dx=5,
                fontSize=14,
            )
            .encode(
                y=alt.Y(
                    "Guard decision:N",
                    sort=guard_order,
                ),
                x=alt.X("Cases:Q"),
                text=alt.Text(
                    "Cases:Q",
                    format=".0f",
                ),
            )
        )

        st.altair_chart(
            guard_bars + guard_labels_chart,
            width="stretch",
        )

        st.dataframe(
            guard_df,
            hide_index=True,
            use_container_width=True,
        )

    with chart_col2:
        st.markdown("#### Candidate eligibility")

        eligibility_long = eligibility_df.melt(
            id_vars=["Candidate"],
            value_vars=["Eligible", "Rejected"],
            var_name="Status",
            value_name="Outputs",
        )

        eligibility_chart = (
            alt.Chart(eligibility_long)
            .mark_bar()
            .encode(
                x=alt.X(
                    "Candidate:N",
                    sort=[
                        "Beam-4",
                        "Greedy",
                        "Concise",
                    ],
                    title=None,
                    axis=alt.Axis(labelAngle=0),
                ),
                y=alt.Y(
                    "Outputs:Q",
                    title="Candidate outputs",
                ),
                color=alt.Color(
                    "Status:N",
                    sort=[
                        "Eligible",
                        "Rejected",
                    ],
                    title=None,
                ),
                order=alt.Order(
                    "Status:N",
                    sort="ascending",
                ),
                tooltip=[
                    alt.Tooltip(
                        "Candidate:N",
                        title="Candidate",
                    ),
                    alt.Tooltip(
                        "Status:N",
                        title="Status",
                    ),
                    alt.Tooltip(
                        "Outputs:Q",
                        title="Outputs",
                    ),
                ],
            )
        )

        st.altair_chart(
            eligibility_chart,
            width="stretch",
        )

        eligibility_display = eligibility_df.copy()

        eligibility_display["Eligibility rate"] = (
            eligibility_display["Eligibility rate"]
            .map(lambda value: f"{value:.2f}%")
        )

        st.dataframe(
            eligibility_display,
            hide_index=True,
            use_container_width=True,
        )

    st.divider()

    consistency_failures = {
        "Numerical precision": sum(
            not bool(
                candidate.get("checks", {}).get(
                    "number_precision",
                    False,
                )
            )
            for candidate in all_candidates
        ),
        "Negation preservation": sum(
            not bool(
                candidate.get("checks", {}).get(
                    "negation_consistency",
                    False,
                )
            )
            for candidate in all_candidates
        ),
        "Uncertainty preservation": sum(
            not bool(
                candidate.get("checks", {}).get(
                    "uncertainty_consistency",
                    False,
                )
            )
            for candidate in all_candidates
        ),
    }

    failure_df = pd.DataFrame(
        {
            "Consistency requirement": list(
                consistency_failures.keys()
            ),
            "Candidate failures": list(
                consistency_failures.values()
            ),
        }
    )

    st.markdown("#### Rule-detected consistency failures")

    failure_maximum = max(
        int(failure_df["Candidate failures"].max()) + 5,
        10,
    )

    failure_order = [
        "Numerical precision",
        "Negation preservation",
        "Uncertainty preservation",
    ]

    failure_bars = (
        alt.Chart(failure_df)
        .mark_bar()
        .encode(
            y=alt.Y(
                "Consistency requirement:N",
                sort=failure_order,
                title=None,
                axis=alt.Axis(labelLimit=220),
            ),
            x=alt.X(
                "Candidate failures:Q",
                title="Rule-detected candidate failures",
                scale=alt.Scale(
                    domain=[0, failure_maximum]
                ),
            ),
            tooltip=[
                alt.Tooltip(
                    "Consistency requirement:N",
                    title="Requirement",
                ),
                alt.Tooltip(
                    "Candidate failures:Q",
                    title="Failures",
                ),
            ],
        )
    )

    failure_labels_chart = (
        alt.Chart(failure_df)
        .mark_text(
            align="left",
            baseline="middle",
            dx=5,
            fontSize=14,
        )
        .encode(
            y=alt.Y(
                "Consistency requirement:N",
                sort=failure_order,
            ),
            x=alt.X("Candidate failures:Q"),
            text=alt.Text(
                "Candidate failures:Q",
                format=".0f",
            ),
        )
    )

    st.altair_chart(
        failure_bars + failure_labels_chart,
        width="stretch",
    )

    st.dataframe(
        failure_df,
        hide_index=True,
        use_container_width=True,
    )

    st.info(
        "These counts represent violations detected by the implemented "
        "number, negation, and uncertainty rules. They are not counts "
        "of manually confirmed factual errors."
    )


with source_tab:
    st.subheader(
        source.get(
            "title",
            "Original biomedical source",
        )
    )

    source_col1, source_col2, source_col3 = st.columns(3)

    source_col1.metric(
        "Document ID",
        source.get("document_id", "—"),
    )

    source_col2.metric(
        "Source words",
        format_integer(
            source.get("word_count")
        ),
    )

    source_col3.metric(
        "Source FKGL",
        format_number(
            source.get("fkgl"),
            2,
        ),
    )

    st.text_area(
        "Original biomedical evidence",
        value=source.get("text", ""),
        height=430,
        disabled=True,
    )


with candidates_tab:
    st.subheader("Generated plain-language candidates")

    st.caption(
        "Eligibility requires numerical precision, "
        "negation preservation, and uncertainty preservation "
        "according to the implemented rule-based checks."
    )

    for candidate in candidates:

        candidate_name = candidate.get(
            "name",
            "Candidate",
        )

        eligible = bool(
            candidate.get(
                "eligible",
                False,
            )
        )

        selected_by_selector = bool(
            candidate.get(
                "selected_by_selector",
                False,
            )
        )

        selected_by_guard = bool(
            candidate.get(
                "selected_by_guard",
                False,
            )
        )

        status_label = (
            "Eligible"
            if eligible
            else "Rejected"
        )

        extra_labels = []

        if selected_by_selector:
            extra_labels.append("Selector choice")

        if selected_by_guard:
            extra_labels.append("Guard choice")

        expander_label = (
            f"{candidate_name} — {status_label}"
        )

        if extra_labels:
            expander_label += (
                " — " + ", ".join(extra_labels)
            )

        with st.expander(
            expander_label,
            expanded=selected_by_guard
            or candidate_name == "Beam-4",
        ):

            if selected_by_guard:
                st.success(
                    "Selected by the frozen "
                    "full-consistency guard."
                )
            elif selected_by_selector:
                st.info(
                    "Selected by the frozen "
                    "reference-free selector."
                )

            score1, score2, score3, score4, score5 = st.columns(5)

            score1.metric(
                "Predicted quality",
                format_number(
                    candidate.get(
                        "predicted_quality"
                    ),
                    3,
                ),
            )

            score2.metric(
                "FKGL",
                format_number(
                    candidate.get("fkgl"),
                    2,
                ),
            )

            score3.metric(
                "Compression",
                format_number(
                    candidate.get(
                        "compression_ratio"
                    ),
                    3,
                ),
            )

            score4.metric(
                "Source similarity",
                format_number(
                    candidate.get(
                        "source_similarity"
                    ),
                    3,
                ),
            )

            score5.metric(
                "Eligibility",
                "Pass" if eligible else "Fail",
            )

            st.text_area(
                f"{candidate_name} output",
                value=candidate.get("text", ""),
                height=280,
                disabled=True,
                key=(
                    f"{selected_id}_"
                    f"{candidate_name}_output"
                ),
            )

            checks = candidate.get(
                "checks",
                {},
            )

            checks_table = pd.DataFrame(
                [
                    {
                        "Consistency check":
                            "Numerical precision",
                        "Status": pass_fail(
                            checks.get(
                                "number_precision",
                                False,
                            )
                        ),
                    },
                    {
                        "Consistency check":
                            "Negation preservation",
                        "Status": pass_fail(
                            checks.get(
                                "negation_consistency",
                                False,
                            )
                        ),
                    },
                    {
                        "Consistency check":
                            "Uncertainty preservation",
                        "Status": pass_fail(
                            checks.get(
                                "uncertainty_consistency",
                                False,
                            )
                        ),
                    },
                ]
            )

            st.table(checks_table)


with decision_tab:
    st.subheader("Final traceable output")

    if mode == "Quality-first":

        beam_candidate = next(
            (
                candidate
                for candidate in candidates
                if candidate.get(
                    "candidate_type"
                ) == "beam4"
            ),
            None,
        )

        if beam_candidate is not None:
            final_text = beam_candidate.get(
                "text",
                "",
            )

            final_label = "Beam-4"
        else:
            final_text = source.get(
                "text",
                "",
            )

            final_label = "Source fallback"

        st.success(
            "Quality-first mode returns the "
            "fixed Beam-4 adaptation."
        )

    else:

        if guard_is_fallback:

            final_text = (
                case.get("guard_output")
                or source.get("text", "")
            )

            final_label = "Source fallback"

            st.warning(
                "No generated candidate satisfied "
                "all locked consistency requirements. "
                "The original biomedical source was returned."
            )

        else:

            final_text = case.get(
                "guard_output",
                "",
            )

            final_label = display_candidate_name(
                guard_decision
            )

            st.success(
                "Reliability-first mode reproduces "
                f"the frozen guard decision: {final_label}."
            )

    st.metric(
        "Selected output",
        final_label,
    )

    st.text_area(
        "Final output",
        value=final_text,
        height=430,
        disabled=True,
    )

    st.caption(
        "The consistency indicators are rule-based. "
        "Passing the checks does not guarantee complete factual correctness."
    )
