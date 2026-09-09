"""Plotly figure builders for the AfyaTrack dashboard.

This layer is deliberately thin: it receives frames that
:mod:`src.analytics` has already scored and aggregated, and its only job is to
choose an encoding. The one exception is the regression overlay on the coverage
scatter, which delegates the fit itself back to
:func:`src.analytics.evaluate_intervention_correlation` so that the line drawn
on screen and the coefficient printed beside it can never disagree.

Colour conventions
------------------
Endemicity strata are an *ordered* scale, not a set of unrelated categories, so
they are encoded on a single-hue navy ramp running light (Low Risk) to dark
(Lake Endemic) rather than on a categorical palette. Every stratum keeps its
step regardless of which strata survive a filter, so narrowing the sidebar
selection never repaints the survivors. Chrome sits on a slate ink scale and
stays recessive against the white clinical surface.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.analytics import (
    ITN_COLUMN,
    PARASITEMIA_COLUMN,
    POPULATION_COLUMN,
    RISK_SCORE_COLUMN,
    ZONE_COLUMN,
    ZONE_SEVERITY_ORDER,
    evaluate_intervention_correlation,
)

COUNTY_COLUMN: Final[str] = "county_name"

SURFACE: Final[str] = "#ffffff"
INK_PRIMARY: Final[str] = "#0f172a"
INK_SECONDARY: Final[str] = "#475569"
INK_MUTED: Final[str] = "#94a3b8"
GRIDLINE: Final[str] = "#e2e8f0"
AXIS_LINE: Final[str] = "#cbd5e1"

FONT_STACK: Final[str] = 'system-ui, -apple-system, "Segoe UI", sans-serif'

#: Ordinal navy ramp, light to dark, keyed to transmission intensity. Steps are
#: spaced so the lightest still clears the 2:1 floor against a white surface.
ZONE_COLORS: Final[dict[str, str]] = {
    "Low Risk": "#86b6ef",
    "Semi-Arid Seasonal": "#5598e7",
    "Highland Epidemic": "#2a78d6",
    "Coast Endemic": "#1c5cab",
    "Lake Endemic": "#0d366b",
}

#: Single-hue default for charts that encode one measure at a time.
PRIMARY_SERIES: Final[str] = "#2a78d6"


def plot_top_risk_counties(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Rank the highest-risk counties as a horizontal bar chart.

    Bar length carries the composite index; fill carries the endemicity
    stratum, which is what turns the ranking into an allocation argument -- a
    highland county appearing among lake-endemic ones is the signal a programme
    officer is looking for.

    Args:
        df: Scored frame containing county, zone, and composite score.
        top_n: Number of counties to display, highest score first.

    Returns:
        A configured Plotly figure. Empty input yields an annotated placeholder.
    """
    if df.empty:
        return _empty_figure("No counties match the current filters")

    ranked = df.nlargest(min(top_n, len(df)), RISK_SCORE_COLUMN)
    # Plotly draws the first category at the bottom of a horizontal axis, so the
    # ascending order here renders as a descending ranking top-to-bottom.
    category_order = ranked.sort_values(RISK_SCORE_COLUMN)[COUNTY_COLUMN].tolist()

    figure = go.Figure()
    for zone in ZONE_SEVERITY_ORDER:
        stratum = ranked[ranked[ZONE_COLUMN] == zone]
        if stratum.empty:
            continue
        figure.add_trace(
            go.Bar(
                x=stratum[RISK_SCORE_COLUMN],
                y=stratum[COUNTY_COLUMN],
                name=zone,
                orientation="h",
                marker=dict(color=ZONE_COLORS[zone], line=dict(width=2, color=SURFACE)),
                text=stratum[RISK_SCORE_COLUMN],
                texttemplate="%{text:.1f}",
                textposition="outside",
                textfont=dict(color=INK_SECONDARY, size=12),
                customdata=stratum[[PARASITEMIA_COLUMN, ITN_COLUMN]].to_numpy(),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Risk index: %{x:.1f} / 100<br>"
                    "Parasitemia: %{customdata[0]:.1f}%<br>"
                    "ITN coverage: %{customdata[1]:.1f}%"
                    "<extra></extra>"
                ),
            )
        )

    figure.update_layout(
        title=dict(text=f"Top {len(ranked)} counties by composite risk index"),
        # One trace per stratum, but no county appears in two traces, so overlay
        # keeps each bar at full band width. Grouped mode would split every
        # category band into one thin slot per stratum.
        barmode="overlay",
        barcornerradius=4,
        bargap=0.28,
        height=max(360, 34 * len(ranked) + 150),
    )
    figure.update_xaxes(
        title_text="Composite risk index (0-100)",
        range=[0, 108],
        showgrid=True,
        gridcolor=GRIDLINE,
        zeroline=False,
    )
    figure.update_yaxes(
        title_text=None,
        categoryorder="array",
        categoryarray=category_order,
        showgrid=False,
    )
    return _apply_clinical_theme(figure)


def plot_itn_vs_parasitemia(df: pd.DataFrame) -> go.Figure:
    """Plot bed-net coverage against parasitemia with a fitted trend line.

    Marker area encodes population so that the counties carrying the most
    people at risk read as the heaviest points, and the OLS line is the same
    fit reported numerically by the analytics layer. The trend line is omitted
    when the selection is too small to support a fit.

    Args:
        df: Validated frame containing coverage, parasitemia, and population.

    Returns:
        A configured Plotly figure. Empty input yields an annotated placeholder.
    """
    if df.empty:
        return _empty_figure("No counties match the current filters")

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=df[ITN_COLUMN],
            y=df[PARASITEMIA_COLUMN],
            mode="markers",
            name="County",
            marker=dict(
                color=PRIMARY_SERIES,
                opacity=0.85,
                sizemode="area",
                size=df[POPULATION_COLUMN],
                sizeref=2.0 * float(df[POPULATION_COLUMN].max()) / (34.0**2),
                sizemin=8,
                line=dict(width=2, color=SURFACE),
            ),
            # `to_numpy` on mixed dtypes yields an object array, which keeps the
            # population an integer; `np.stack` would upcast it to a string and
            # break the numeric hover format.
            customdata=df[[COUNTY_COLUMN, ZONE_COLUMN, POPULATION_COLUMN]].to_numpy(),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "%{customdata[1]}<br>"
                "ITN coverage: %{x:.1f}%<br>"
                "Parasitemia: %{y:.1f}%<br>"
                "Population: %{customdata[2]:,.0f}"
                "<extra></extra>"
            ),
        )
    )

    trend = _fit_trend_line(df)
    if trend is not None:
        x_fit, y_fit, label = trend
        figure.add_trace(
            go.Scatter(
                x=x_fit,
                y=y_fit,
                mode="lines",
                name=label,
                line=dict(color=INK_SECONDARY, width=2, dash="dash"),
                hoverinfo="skip",
            )
        )

    figure.update_layout(
        title=dict(text="ITN coverage against RDT parasitemia, by county"),
        height=460,
        hovermode="closest",
    )
    figure.update_xaxes(
        title_text="ITN coverage (%)", showgrid=True, gridcolor=GRIDLINE, zeroline=False
    )
    figure.update_yaxes(
        title_text="Parasitemia, RDT positive (%)",
        showgrid=True,
        gridcolor=GRIDLINE,
        zeroline=False,
    )
    return _apply_clinical_theme(figure)


def plot_endemicity_breakdown(summary: pd.DataFrame) -> go.Figure:
    """Compare strata on risk and on parasitemia as paired small multiples.

    The two measures sit on different scales, so they get their own panels
    against a shared stratum axis rather than a second y-axis, which would let
    the panel geometry imply a crossover that the data does not contain.

    Args:
        summary: Output of :func:`src.analytics.get_strata_summary`.

    Returns:
        A configured Plotly figure. Empty input yields an annotated placeholder.
    """
    if summary.empty:
        return _empty_figure("No strata match the current filters")

    ordered = summary.sort_values("mean_risk_index")
    zones = ordered[ZONE_COLUMN].tolist()
    colors = [ZONE_COLORS.get(zone, PRIMARY_SERIES) for zone in zones]

    figure = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.10,
        subplot_titles=("Mean composite risk index", "Mean parasitemia (%)"),
    )
    panels = (
        ("mean_risk_index", "%{x:.1f}", "Risk index: %{x:.1f} / 100"),
        ("mean_parasitemia_pct", "%{x:.1f}%", "Parasitemia: %{x:.1f}%"),
    )
    for column_index, (measure, text_format, hover_line) in enumerate(panels, start=1):
        figure.add_trace(
            go.Bar(
                x=ordered[measure],
                y=zones,
                orientation="h",
                marker=dict(color=colors, line=dict(width=2, color=SURFACE)),
                text=ordered[measure],
                texttemplate=text_format,
                textposition="outside",
                textfont=dict(color=INK_SECONDARY, size=12),
                customdata=ordered[["county_count", "population_at_risk"]].to_numpy(),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    f"{hover_line}<br>"
                    "Counties: %{customdata[0]}<br>"
                    "Population at risk: %{customdata[1]:,.0f}"
                    "<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1,
            col=column_index,
        )

    figure.update_layout(
        title=dict(text="Endemicity stratum comparison"),
        barcornerradius=4,
        bargap=0.35,
        height=max(320, 54 * len(zones) + 160),
    )
    figure.update_xaxes(showgrid=True, gridcolor=GRIDLINE, zeroline=False, rangemode="tozero")
    figure.update_yaxes(showgrid=False)
    # Headroom so the outside value labels are not clipped by the panel edge.
    for column_index, (measure, _, _) in enumerate(panels, start=1):
        figure.update_xaxes(
            range=[0, float(ordered[measure].max()) * 1.22], row=1, col=column_index
        )
    return _apply_clinical_theme(figure)


def _fit_trend_line(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, str] | None:
    """Return the OLS overlay for the coverage scatter, or None if unfittable.

    Reuses the analytics fit rather than repeating ``polyfit`` here, so the
    drawn line and the reported slope are guaranteed to be the same model.
    """
    try:
        fit = evaluate_intervention_correlation(df)
    except (ValueError, KeyError):
        return None

    x_fit = np.linspace(float(df[ITN_COLUMN].min()), float(df[ITN_COLUMN].max()), num=2)
    y_fit = float(fit["slope"]) * x_fit + float(fit["intercept"])
    label = f"OLS fit (r = {float(fit['pearson_r']):.2f})"
    return x_fit, y_fit, label


def _apply_clinical_theme(figure: go.Figure) -> go.Figure:
    """Apply the shared white/navy/slate chrome to a figure."""
    figure.update_layout(
        template="plotly_white",
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT_STACK, size=13, color=INK_SECONDARY),
        title=dict(font=dict(size=16, color=INK_PRIMARY), x=0.0, xanchor="left", y=0.96),
        margin=dict(l=8, r=24, t=64, b=48),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
            title_text=None,
            font=dict(size=12),
        ),
        hoverlabel=dict(
            bgcolor=SURFACE,
            bordercolor=AXIS_LINE,
            font=dict(family=FONT_STACK, size=12, color=INK_PRIMARY),
        ),
    )
    figure.update_xaxes(
        linecolor=AXIS_LINE,
        tickfont=dict(color=INK_MUTED, size=12),
        title_font=dict(color=INK_SECONDARY, size=12),
    )
    figure.update_yaxes(
        linecolor=AXIS_LINE,
        tickfont=dict(color=INK_SECONDARY, size=12),
        title_font=dict(color=INK_SECONDARY, size=12),
    )
    for annotation in figure.layout.annotations:
        annotation.font.update(size=13, color=INK_PRIMARY)
    return figure


def _empty_figure(message: str) -> go.Figure:
    """Render an explicit empty state instead of a blank plotting area."""
    figure = go.Figure()
    figure.add_annotation(
        text=message,
        showarrow=False,
        font=dict(family=FONT_STACK, size=14, color=INK_MUTED),
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
    )
    figure.update_layout(
        height=260,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=8, r=8, t=8, b=8),
    )
    return figure
