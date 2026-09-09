"""Analytical model layer: risk stratification and intervention evaluation.

Every function here takes a frame that has already passed
:func:`src.ingestion.validate_surveillance_frame` and returns a new frame or a
plain dict. Nothing mutates its argument, and nothing reads from disk, so the
whole layer is deterministic and directly unit-testable.

The composite risk index is the platform's central artefact. It exists to answer
one operational question -- *given a finite commodity budget, which counties do
we treat first?* -- so it deliberately blends burden with intervention gap: a
county already at high coverage scores lower than an equally-infected county
where nets have not reached households.
"""

from __future__ import annotations

import logging
from typing import Final

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

PARASITEMIA_COLUMN: Final[str] = "parasitemia_rate_rdt_pct"
INCIDENCE_COLUMN: Final[str] = "confirmed_cases_per_1000"
ITN_COLUMN: Final[str] = "itn_coverage_pct"
ZONE_COLUMN: Final[str] = "endemicity_zone"
POPULATION_COLUMN: Final[str] = "population"

PARASITEMIA_INDEX_COLUMN: Final[str] = "parasitemia_index"
INCIDENCE_INDEX_COLUMN: Final[str] = "incidence_index"
NET_GAP_INDEX_COLUMN: Final[str] = "net_gap_index"
RISK_SCORE_COLUMN: Final[str] = "composite_risk_score"
RISK_TIER_COLUMN: Final[str] = "risk_tier"

#: Component weights for the composite index. Parasitemia carries the plurality
#: because RDT prevalence is the least reporting-dependent of the three signals;
#: confirmed incidence is weighted lower precisely because it tracks
#: health-facility access as much as it tracks transmission. The net gap enters
#: at 20% as an actionability term rather than a burden term.
RISK_WEIGHTS: Final[dict[str, float]] = {
    PARASITEMIA_INDEX_COLUMN: 0.50,
    INCIDENCE_INDEX_COLUMN: 0.30,
    NET_GAP_INDEX_COLUMN: 0.20,
}

#: Tier cut points on the 0-100 index, used for triage colouring and for the
#: prioritisation table. Bins are left-open, right-closed except at the floor.
RISK_TIER_BOUNDS: Final[tuple[float, ...]] = (0.0, 25.0, 50.0, 75.0, 100.0)
RISK_TIER_LABELS: Final[tuple[str, ...]] = ("Low", "Moderate", "High", "Critical")

#: Strata ordered by transmission intensity, most intense first. Used to keep
#: group ordering and chart colour assignment stable across filtered views.
ZONE_SEVERITY_ORDER: Final[tuple[str, ...]] = (
    "Lake Endemic",
    "Coast Endemic",
    "Highland Epidemic",
    "Semi-Arid Seasonal",
    "Low Risk",
)

#: Pearson correlation and the OLS slope are undefined below three points, and
#: a two-point fit is a tautology rather than evidence.
MIN_CORRELATION_SAMPLE: Final[int] = 3


def calculate_composite_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """Attach a normalised 0-100 composite risk index to each county.

    Each of the three signals is min-max rescaled to 0-100 across the counties
    present in ``df``, then combined under :data:`RISK_WEIGHTS`. Because the
    weights sum to 1 and each component is bounded to 0-100, the composite is
    bounded to 0-100 by construction.

    The scaling is **cohort-relative**: a county's score expresses its standing
    against the other counties in the frame, not against an absolute scale.
    Callers presenting a filtered view should therefore score the full national
    cohort first and filter afterwards, otherwise scores shift as the user
    changes filters.

    Args:
        df: Validated surveillance frame. Not mutated.

    Returns:
        A copy of ``df`` with the three component indices, the composite score,
        and an ordered categorical risk tier appended.

    Raises:
        KeyError: If a column the model depends on is absent.
    """
    _require_columns(df, (PARASITEMIA_COLUMN, INCIDENCE_COLUMN, ITN_COLUMN))

    scored = df.copy()
    scored[PARASITEMIA_INDEX_COLUMN] = _rescale(scored[PARASITEMIA_COLUMN], PARASITEMIA_COLUMN)
    scored[INCIDENCE_INDEX_COLUMN] = _rescale(scored[INCIDENCE_COLUMN], INCIDENCE_COLUMN)

    # Risk rises as coverage falls, so the protective measure is inverted into
    # an unmet-need term before scaling.
    net_gap = 100.0 - scored[ITN_COLUMN]
    scored[NET_GAP_INDEX_COLUMN] = _rescale(net_gap, "itn_coverage_gap")

    scored[RISK_SCORE_COLUMN] = sum(
        scored[component] * weight for component, weight in RISK_WEIGHTS.items()
    )
    # Guard against float accumulation nudging a boundary value past 100.
    scored[RISK_SCORE_COLUMN] = scored[RISK_SCORE_COLUMN].clip(lower=0.0, upper=100.0)
    scored[RISK_TIER_COLUMN] = _assign_risk_tier(scored[RISK_SCORE_COLUMN])

    LOGGER.debug(
        "Scored %d counties; index spans %.2f-%.2f",
        len(scored),
        scored[RISK_SCORE_COLUMN].min(),
        scored[RISK_SCORE_COLUMN].max(),
    )
    return scored


def evaluate_intervention_correlation(df: pd.DataFrame) -> dict[str, float | int | str]:
    """Quantify the association between bed-net coverage and parasitemia.

    Fits an ordinary least-squares line of parasitemia on ITN coverage and
    reports the Pearson coefficient alongside it. The slope is the operationally
    useful number: it estimates the percentage-point change in RDT positivity
    associated with each additional percentage point of coverage.

    This is an ecological, cross-sectional association and is reported as such.
    Counties with high coverage are also counties targeted *because* they were
    high burden, so the coefficient must not be read as an effect size.

    Args:
        df: Validated surveillance frame containing coverage and parasitemia.

    Returns:
        A dict with ``pearson_r``, ``r_squared``, ``slope``, ``intercept``,
        ``t_statistic``, ``n_counties``, and a human-readable ``direction``.

    Raises:
        KeyError: If a required column is absent.
        ValueError: If fewer than :data:`MIN_CORRELATION_SAMPLE` counties are
            supplied, or if either variable has zero variance.
    """
    _require_columns(df, (ITN_COLUMN, PARASITEMIA_COLUMN))

    coverage = df[ITN_COLUMN].to_numpy(dtype=float)
    parasitemia = df[PARASITEMIA_COLUMN].to_numpy(dtype=float)
    n_counties = coverage.size

    if n_counties < MIN_CORRELATION_SAMPLE:
        raise ValueError(
            f"Correlation requires at least {MIN_CORRELATION_SAMPLE} counties; "
            f"received {n_counties}"
        )
    if coverage.std() == 0.0 or parasitemia.std() == 0.0:
        raise ValueError(
            "Correlation is undefined when either coverage or parasitemia has "
            "zero variance across the selected counties"
        )

    pearson_r = float(np.corrcoef(coverage, parasitemia)[0, 1])
    slope, intercept = (float(value) for value in np.polyfit(coverage, parasitemia, deg=1))

    # t = r * sqrt((n - 2) / (1 - r^2)) on n-2 degrees of freedom. Computed
    # directly rather than pulling in SciPy for a single statistic.
    r_squared = pearson_r**2
    denominator = 1.0 - r_squared
    t_statistic = (
        float("inf")
        if denominator <= 0.0
        else float(pearson_r * np.sqrt((n_counties - 2) / denominator))
    )

    return {
        "pearson_r": pearson_r,
        "r_squared": r_squared,
        "slope": slope,
        "intercept": intercept,
        "t_statistic": t_statistic,
        "n_counties": int(n_counties),
        "direction": _describe_association(pearson_r),
    }


def get_strata_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate county records into endemicity-stratum profiles.

    Micro-stratification is the unit at which Kenya's national programme plans
    commodity allocation, so this is the frame that drives zone-level targeting
    decisions in the dashboard.

    Args:
        df: Validated surveillance frame. If it does not already carry a
            composite score, one is computed over the frame as given.

    Returns:
        One row per stratum present in ``df``, sorted by mean risk descending,
        with columns ``endemicity_zone``, ``county_count``,
        ``population_at_risk``, ``mean_parasitemia_pct``,
        ``mean_itn_coverage_pct``, and ``mean_risk_index``.

    Raises:
        KeyError: If a required column is absent.
    """
    _require_columns(df, (ZONE_COLUMN, POPULATION_COLUMN, PARASITEMIA_COLUMN, ITN_COLUMN))

    scored = df if RISK_SCORE_COLUMN in df.columns else calculate_composite_risk_score(df)

    grouped = scored.groupby(ZONE_COLUMN, observed=True, dropna=True)
    # `size` is taken off the population column rather than the group key,
    # which pandas excludes from the aggregation frame.
    summary = grouped.agg(
        county_count=(POPULATION_COLUMN, "size"),
        population_at_risk=(POPULATION_COLUMN, "sum"),
        mean_parasitemia_pct=(PARASITEMIA_COLUMN, "mean"),
        mean_itn_coverage_pct=(ITN_COLUMN, "mean"),
        mean_risk_index=(RISK_SCORE_COLUMN, "mean"),
    )

    summary = summary.reset_index()
    summary["population_at_risk"] = summary["population_at_risk"].astype("int64")
    return summary.sort_values("mean_risk_index", ascending=False).reset_index(drop=True)


def order_zones(zones: pd.Series) -> pd.Categorical:
    """Cast a zone column to a categorical ordered by transmission intensity.

    Keeps axis ordering and colour assignment identical between the national
    view and any filtered subset, so a stratum never changes colour when the
    user narrows the selection.
    """
    present = [zone for zone in ZONE_SEVERITY_ORDER if zone in set(zones)]
    return pd.Categorical(zones, categories=present, ordered=True)


def _rescale(series: pd.Series, label: str) -> pd.Series:
    """Min-max rescale a series onto 0-100.

    A series with no spread carries no ranking information, so it contributes
    nothing to the composite rather than an arbitrary constant. This is reached
    only for degenerate cohorts -- a single county, or identical values -- and
    is logged when it happens.
    """
    minimum = float(series.min())
    maximum = float(series.max())
    spread = maximum - minimum

    if spread <= 0.0:
        LOGGER.warning(
            "Component '%s' has no spread across %d row(s); contributing 0 to the index",
            label,
            len(series),
        )
        return pd.Series(0.0, index=series.index, dtype=float)

    return (series - minimum) / spread * 100.0


def _assign_risk_tier(scores: pd.Series) -> pd.Series:
    """Bucket composite scores into ordered triage tiers."""
    tiers = pd.cut(
        scores,
        bins=list(RISK_TIER_BOUNDS),
        labels=list(RISK_TIER_LABELS),
        include_lowest=True,
        ordered=True,
    )
    return tiers.astype(pd.CategoricalDtype(categories=list(RISK_TIER_LABELS), ordered=True))


def _describe_association(pearson_r: float) -> str:
    """Render a Pearson coefficient as a direction and strength phrase."""
    magnitude = abs(pearson_r)
    if magnitude < 0.20:
        strength = "negligible"
    elif magnitude < 0.40:
        strength = "weak"
    elif magnitude < 0.60:
        strength = "moderate"
    elif magnitude < 0.80:
        strength = "strong"
    else:
        strength = "very strong"

    if magnitude < 0.20:
        return f"{strength} association"
    direction = "positive" if pearson_r > 0 else "inverse"
    return f"{strength} {direction} association"


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> None:
    """Raise a single, complete KeyError when expected columns are absent."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Frame is missing required column(s): {', '.join(missing)}")
