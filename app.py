"""AfyaTrack dashboard: subnational malaria surveillance and prioritisation.

Entry point for the Streamlit presentation layer. This module owns layout,
filter state, and formatting only -- every number it renders comes from
:mod:`src.analytics`, and every figure from :mod:`src.visualization`.

One ordering decision matters analytically and is enforced here rather than in
the analytics layer: the composite risk index is computed **once over the full
47-county national cohort**, and sidebar filters are applied to the already
scored frame. Scoring after filtering would rescale the index against whatever
subset the user happened to select, so a county's risk would change as the user
moved a slider. Ranks stay national; the view narrows.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Final

import pandas as pd
import streamlit as st

from src.analytics import (
    ITN_COLUMN,
    PARASITEMIA_COLUMN,
    POPULATION_COLUMN,
    RISK_SCORE_COLUMN,
    RISK_TIER_COLUMN,
    RISK_WEIGHTS,
    ZONE_COLUMN,
    ZONE_SEVERITY_ORDER,
    calculate_composite_risk_score,
    evaluate_intervention_correlation,
    get_strata_summary,
)
from src.ingestion import load_surveillance_data
from src.visualization import (
    plot_endemicity_breakdown,
    plot_itn_vs_parasitemia,
    plot_top_risk_counties,
)

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent
DEFAULT_DATA_PATH: Final[Path] = PROJECT_ROOT / "data" / "kenya_malaria_surveillance.csv"
COUNTY_COLUMN: Final[str] = "county_name"

CLINICAL_CSS: Final[str] = """
<style>
  .stApp { background-color: #f8fafc; }
  .block-container { padding-top: 2.4rem; max-width: 1320px; }

  .afya-masthead {
    border-left: 4px solid #0d366b;
    padding: 0.1rem 0 0.1rem 1rem;
    margin-bottom: 0.35rem;
  }
  .afya-masthead h1 {
    font-size: 1.65rem; font-weight: 650; color: #0f172a;
    margin: 0; letter-spacing: -0.015em;
  }
  .afya-masthead p {
    font-size: 0.92rem; color: #475569; margin: 0.3rem 0 0 0;
  }

  .afya-kpi {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1rem 1.1rem;
    height: 100%;
  }
  .afya-kpi .afya-kpi-label {
    font-size: 0.74rem; font-weight: 600; letter-spacing: 0.07em;
    text-transform: uppercase; color: #64748b; margin-bottom: 0.45rem;
  }
  .afya-kpi .afya-kpi-value {
    font-size: 1.72rem; font-weight: 620; color: #0f172a; line-height: 1.15;
  }
  .afya-kpi .afya-kpi-context {
    font-size: 0.8rem; color: #64748b; margin-top: 0.3rem;
  }

  .afya-note {
    background: #ffffff; border: 1px solid #e2e8f0;
    border-left: 3px solid #2a78d6; border-radius: 8px;
    padding: 0.85rem 1rem; font-size: 0.88rem; color: #475569;
  }
  .afya-formula {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px;
    padding: 1rem 1.15rem; font-family: ui-monospace, "Cascadia Code", monospace;
    font-size: 0.86rem; color: #0f172a; line-height: 1.7;
  }
  .stTabs [data-baseweb="tab-list"] { gap: 1.6rem; border-bottom: 1px solid #e2e8f0; }
  .stTabs [data-baseweb="tab"] { font-size: 0.92rem; font-weight: 550; color: #475569; }
  section[data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e2e8f0; }
</style>
"""


@st.cache_data(show_spinner="Loading national surveillance extract...")
def load_national_cohort(data_path: str) -> pd.DataFrame:
    """Ingest, validate, and score the full national cohort.

    Cached because the result is identical for every session and every filter
    combination; the sidebar narrows this frame rather than recomputing it.

    Args:
        data_path: Filesystem path to the surveillance CSV extract.

    Returns:
        The validated 47-county frame with composite risk columns attached.
    """
    surveillance = load_surveillance_data(data_path)
    return calculate_composite_risk_score(surveillance)


def render_sidebar(cohort: pd.DataFrame) -> pd.DataFrame:
    """Render the filter controls and return the resulting county selection.

    Args:
        cohort: The scored national cohort.

    Returns:
        The subset of ``cohort`` matching the active filters.
    """
    st.sidebar.markdown("### Cohort filters")
    st.sidebar.caption(
        "Filters narrow the view only. Risk indices remain scored against all "
        "47 counties so rankings stay nationally comparable."
    )

    available_zones = [zone for zone in ZONE_SEVERITY_ORDER if zone in set(cohort[ZONE_COLUMN])]
    selected_zones = st.sidebar.multiselect(
        "Endemicity zone",
        options=available_zones,
        default=available_zones,
        help="Kenya Malaria Strategy transmission strata, ordered by intensity.",
    )

    # Bounds are snapped outward to whole steps so the slider grid lands exactly
    # on both endpoints. Left unrounded, the largest county sits between two
    # stops and drops out of the selection as soon as the handle is nudged.
    step = 25_000
    population_floor = int(cohort[POPULATION_COLUMN].min()) // step * step
    population_ceiling = -(-int(cohort[POPULATION_COLUMN].max()) // step) * step
    selected_population = st.sidebar.slider(
        "County population",
        min_value=population_floor,
        max_value=population_ceiling,
        value=(population_floor, population_ceiling),
        step=step,
        format="%d",
        help="Restrict to counties within a population band.",
    )

    selection = cohort[
        cohort[ZONE_COLUMN].isin(selected_zones)
        & cohort[POPULATION_COLUMN].between(*selected_population)
    ]

    st.sidebar.divider()
    st.sidebar.metric("Counties in view", f"{len(selection)} of {len(cohort)}")
    return selection


def render_kpi_row(selection: pd.DataFrame) -> None:
    """Render the four headline indicators for the current selection."""
    if selection.empty:
        st.warning("No counties match the current filters. Widen the selection to continue.")
        return

    population_at_risk = int(selection[POPULATION_COLUMN].sum())
    # Population-weighted rather than a flat county mean: an unweighted average
    # lets Lamu (161k residents) and Nairobi (4.9m) move the national figure by
    # the same amount, which misstates the true burden.
    weighted_parasitemia = float(
        (selection[PARASITEMIA_COLUMN] * selection[POPULATION_COLUMN]).sum()
        / selection[POPULATION_COLUMN].sum()
    )
    highest_burden = selection.loc[selection[RISK_SCORE_COLUMN].idxmax()]
    median_coverage = float(selection[ITN_COLUMN].median())

    cards = (
        (
            "Population at risk",
            f"{population_at_risk / 1_000_000:.2f}M",
            f"across {len(selection)} counties",
        ),
        (
            "Avg parasitemia",
            f"{weighted_parasitemia:.2f}%",
            "RDT positivity, population-weighted",
        ),
        (
            "Highest burden county",
            str(highest_burden[COUNTY_COLUMN]),
            f"risk index {highest_burden[RISK_SCORE_COLUMN]:.1f} / 100",
        ),
        (
            "Median ITN coverage",
            f"{median_coverage:.1f}%",
            "household bed-net usage",
        ),
    )

    for column, (label, value, context) in zip(st.columns(4), cards):
        column.markdown(
            f'<div class="afya-kpi">'
            f'<div class="afya-kpi-label">{label}</div>'
            f'<div class="afya-kpi-value">{value}</div>'
            f'<div class="afya-kpi-context">{context}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )


def render_surveillance_tab(selection: pd.DataFrame) -> None:
    """Render the county prioritisation matrix and the top-risk ranking."""
    st.markdown("#### County surveillance and prioritisation matrix")
    st.caption(
        "Sorted by composite risk index. Tiers are fixed cut points on the "
        "index, so a county's tier does not move when filters change."
    )

    if selection.empty:
        st.info("No counties to display.")
        return

    table = selection.sort_values(RISK_SCORE_COLUMN, ascending=False)
    # The tier is an ordered categorical; Streamlit's TextColumn expects a
    # plain string, so it is flattened for display only.
    table = table.assign(**{RISK_TIER_COLUMN: table[RISK_TIER_COLUMN].astype(str)})
    display_columns = [
        COUNTY_COLUMN,
        ZONE_COLUMN,
        POPULATION_COLUMN,
        PARASITEMIA_COLUMN,
        "confirmed_cases_per_1000",
        ITN_COLUMN,
        RISK_SCORE_COLUMN,
        RISK_TIER_COLUMN,
    ]
    st.dataframe(
        table[display_columns],
        hide_index=True,
        use_container_width=True,
        height=396,
        column_config={
            COUNTY_COLUMN: st.column_config.TextColumn("County"),
            ZONE_COLUMN: st.column_config.TextColumn("Endemicity zone"),
            POPULATION_COLUMN: st.column_config.NumberColumn("Population", format="%d"),
            PARASITEMIA_COLUMN: st.column_config.NumberColumn("Parasitemia %", format="%.1f"),
            "confirmed_cases_per_1000": st.column_config.NumberColumn(
                "Cases / 1,000", format="%.1f"
            ),
            ITN_COLUMN: st.column_config.NumberColumn("ITN %", format="%.1f"),
            RISK_SCORE_COLUMN: st.column_config.ProgressColumn(
                "Risk index", format="%.1f", min_value=0, max_value=100
            ),
            RISK_TIER_COLUMN: st.column_config.TextColumn("Tier"),
        },
    )
    st.download_button(
        "Download prioritisation matrix (CSV)",
        data=table[display_columns].to_csv(index=False).encode("utf-8"),
        file_name="afyatrack_prioritisation_matrix.csv",
        mime="text/csv",
    )

    st.divider()
    st.plotly_chart(plot_top_risk_counties(selection), use_container_width=True)


def render_intervention_tab(selection: pd.DataFrame) -> None:
    """Render the ITN-parasitemia association and the stratum aggregates."""
    st.markdown("#### Intervention analytics")

    if selection.empty:
        st.info("No counties to analyse.")
        return

    try:
        fit = evaluate_intervention_correlation(selection)
    except ValueError as exc:
        fit = None
        st.info(f"Correlation unavailable for this selection: {exc}")

    if fit is not None:
        statistics = st.columns(3)
        statistics[0].metric("Pearson r", f"{float(fit['pearson_r']):+.3f}", str(fit["direction"]))
        statistics[1].metric("R-squared", f"{float(fit['r_squared']):.3f}", "variance explained")
        statistics[2].metric(
            "OLS slope",
            f"{float(fit['slope']):+.3f}",
            "pp parasitemia per pp coverage",
        )
        st.markdown(
            '<div class="afya-note"><b>Interpretation.</b> This is a '
            "cross-sectional ecological association across counties, not an "
            "effect estimate. Coverage is highest precisely where burden is "
            "highest, because mass net campaigns are targeted at endemic "
            "strata. The positive coefficient therefore reflects programme "
            "targeting, and reading it as nets increasing transmission would "
            "invert the causal direction.</div>",
            unsafe_allow_html=True,
        )
        st.write("")

    st.plotly_chart(plot_itn_vs_parasitemia(selection), use_container_width=True)

    st.divider()
    st.markdown("#### Endemicity stratum aggregates")
    summary = get_strata_summary(selection)
    st.dataframe(
        summary,
        hide_index=True,
        use_container_width=True,
        column_config={
            ZONE_COLUMN: st.column_config.TextColumn("Endemicity zone"),
            "county_count": st.column_config.NumberColumn("Counties", format="%d"),
            "population_at_risk": st.column_config.NumberColumn(
                "Population at risk", format="%d"
            ),
            "mean_parasitemia_pct": st.column_config.NumberColumn(
                "Mean parasitemia %", format="%.2f"
            ),
            "mean_itn_coverage_pct": st.column_config.NumberColumn(
                "Mean ITN %", format="%.1f"
            ),
            "mean_risk_index": st.column_config.NumberColumn(
                "Mean risk index", format="%.1f"
            ),
        },
    )
    st.plotly_chart(plot_endemicity_breakdown(summary), use_container_width=True)


def render_methodology_tab() -> None:
    """Document the pipeline, the risk model, and the delivery process."""
    st.markdown("#### Pipeline architecture and methodology")

    st.markdown("##### 1. Data ingestion and schema enforcement")
    st.markdown(
        "`src.ingestion.load_surveillance_data` is the only entry point for raw "
        "extracts. It separates two failure classes deliberately. **Contract "
        "defects** -- a missing column, an unparsable file, an extract with no "
        "usable rows -- raise `SchemaError` and stop the pipeline, because they "
        "mean the upstream export is wrong. **Row defects** -- a blank cell, a "
        "parasitemia rate above 100%, an unrecognised endemicity label, a "
        "duplicated county -- drop that county and emit a `logging` warning "
        "naming it, so one malformed district never costs the other 46."
    )
    st.markdown(
        "Numeric bounds are plausibility limits, not observed extremes: they "
        "exist to catch unit errors such as a rate exported on a 0-1 scale, "
        "not to second-guess an unusual county."
    )

    st.markdown("##### 2. Composite risk index")
    st.markdown(
        "Each signal is min-max rescaled to 0-100 across the national cohort, "
        "then combined under fixed weights. Because the weights sum to 1 and "
        "each component is bounded, the composite is bounded to 0-100 by "
        "construction -- a property asserted directly in `tests/test_analytics.py`."
    )
    st.markdown(
        '<div class="afya-formula">'
        "norm(x) = (x - min(x)) / (max(x) - min(x)) &times; 100<br><br>"
        f"Risk = {RISK_WEIGHTS['parasitemia_index']:.2f} &times; "
        "norm(parasitemia_rate_rdt_pct)<br>"
        f"&nbsp;&nbsp;&nbsp;&nbsp;+ {RISK_WEIGHTS['incidence_index']:.2f} &times; "
        "norm(confirmed_cases_per_1000)<br>"
        f"&nbsp;&nbsp;&nbsp;&nbsp;+ {RISK_WEIGHTS['net_gap_index']:.2f} &times; "
        "norm(100 - itn_coverage_pct)"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "**Why these weights.** Parasitemia carries the plurality because RDT "
        "prevalence is the least reporting-dependent of the three signals. "
        "Confirmed incidence is weighted lower precisely because it tracks "
        "health-facility access as much as transmission. The net gap enters at "
        "20% as an *actionability* term rather than a burden term: between two "
        "equally infected counties, the one where nets have not reached "
        "households is the one where a delivered commodity changes an outcome."
    )
    st.markdown(
        "**Known limitations.** Min-max scaling is cohort-relative, so the "
        "index ranks counties against each other rather than against an "
        "absolute threshold. The dashboard scores the full national cohort "
        "before applying filters to keep rankings stable across views; a "
        "cross-year comparison would require fixed reference bounds instead."
    )
    st.markdown(
        "Separately, the actionability term puts a **floor** under counties "
        "that have almost no transmission but weak coverage: a highland county "
        "at 0.2% parasitemia still earns most of the 20-point net-gap "
        "allocation, and can therefore outrank a county with ten times its "
        "burden. The fixed tier cut points contain this -- everything under 25 "
        "reads *Low* regardless -- but the raw index should be read as a "
        "prioritisation ranking, not as a burden measure, at the bottom of its "
        "range."
    )

    st.markdown("##### 3. Architecture")
    st.markdown(
        "```\n"
        "data/*.csv\n"
        "  -> src.ingestion      schema contract, coercion, row quarantine\n"
        "  -> src.analytics      risk index, OLS fit, stratum aggregation\n"
        "  -> src.visualization  Plotly encodings (no analytical logic)\n"
        "  -> app.py             layout, filter state, formatting\n"
        "```"
    )
    st.markdown(
        "The dependency arrow runs one way. `src.analytics` never reads from "
        "disk and never imports Streamlit, so the entire analytical surface is "
        "testable without a browser or a fixture file. `src.visualization` "
        "delegates its regression overlay back to the analytics fit rather than "
        "repeating the calculation, so the drawn line and the printed "
        "coefficient cannot diverge."
    )

    st.markdown("##### 4. SDLC and delivery")
    st.markdown(
        "- **Quality gates.** `flake8` (100-column limit) and `pytest` run on "
        "every push and pull request via `.github/workflows/ci.yml`, across "
        "Python 3.11 and 3.12.\n"
        "- **Test strategy.** Ingestion tests assert the contract boundary "
        "(valid load, missing column, out-of-range quarantine). Analytics "
        "tests assert mathematical invariants -- index bounds, weight "
        "composition, monotonicity, and aggregate schemas.\n"
        "- **Virtualisation.** A multi-stage `python:3.11-slim` image installs "
        "dependencies into a wheel layer, then copies only the runtime "
        "artefacts into a final image that runs as an unprivileged user with a "
        "container healthcheck on the Streamlit endpoint.\n"
        "- **Reproducibility.** `make install / test / lint / run / docker-up` "
        "give identical entry points locally and in CI."
    )


def main() -> None:
    """Compose and render the dashboard."""
    st.set_page_config(
        page_title="AfyaTrack | Malaria Surveillance",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CLINICAL_CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="afya-masthead">'
        "<h1>AfyaTrack</h1>"
        "<p>Subnational malaria surveillance and analytical modeling platform "
        "&mdash; 47-county micro-stratification, Kenya</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    data_path = os.environ.get("AFYATRACK_DATA_PATH", str(DEFAULT_DATA_PATH))
    try:
        cohort = load_national_cohort(data_path)
    except (FileNotFoundError, ValueError) as exc:
        st.error(f"Surveillance extract could not be loaded: {exc}")
        st.caption(f"Resolved path: `{data_path}`")
        st.stop()
        return

    selection = render_sidebar(cohort)
    st.write("")
    render_kpi_row(selection)
    st.write("")

    surveillance_tab, intervention_tab, methodology_tab = st.tabs(
        ["County surveillance", "Intervention analytics", "Pipeline & methodology"]
    )
    with surveillance_tab:
        render_surveillance_tab(selection)
    with intervention_tab:
        render_intervention_tab(selection)
    with methodology_tab:
        render_methodology_tab()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    main()
