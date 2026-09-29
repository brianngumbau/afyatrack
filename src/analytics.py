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
import math
from typing import Final

import numpy as np
import pandas as pd

from src.ingestion import ENDEMICITY_ZONES

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
#: Aliased to the ingestion contract so the two lists cannot drift apart.
ZONE_SEVERITY_ORDER: Final[tuple[str, ...]] = ENDEMICITY_ZONES

#: Pearson correlation and the OLS slope are undefined below three points, and
#: a two-point fit is a tautology rather than evidence.
MIN_CORRELATION_SAMPLE: Final[int] = 3

#: Operational constant: Kenya Ministry of Health planning standard (1 ITN per 1.8 persons)
PERSONS_PER_ITN: Final[float] = 1.8
MAX_FEASIBLE_ITN_COVERAGE: Final[float] = 98.0


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
        math.copysign(float("inf"), pearson_r)
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
        ``weighted_parasitemia_pct``, ``mean_itn_coverage_pct``, and
        ``mean_risk_index``. The weighted rate is population-weighted, matching
        the dashboard's headline figure; the plain mean is a county average.

    Raises:
        KeyError: If a required column is absent.
    """
    _require_columns(df, (ZONE_COLUMN, POPULATION_COLUMN, PARASITEMIA_COLUMN, ITN_COLUMN))

    scored = df if RISK_SCORE_COLUMN in df.columns else calculate_composite_risk_score(df)

    # Numerator for the population-weighted rate, summed per stratum below.
    scored = scored.assign(
        _parasitemia_person_pct=scored[PARASITEMIA_COLUMN] * scored[POPULATION_COLUMN]
    )
    grouped = scored.groupby(ZONE_COLUMN, observed=True, dropna=True)
    # `size` is taken off the population column rather than the group key,
    # which pandas excludes from the aggregation frame.
    summary = grouped.agg(
        county_count=(POPULATION_COLUMN, "size"),
        population_at_risk=(POPULATION_COLUMN, "sum"),
        mean_parasitemia_pct=(PARASITEMIA_COLUMN, "mean"),
        parasitemia_person_pct=("_parasitemia_person_pct", "sum"),
        mean_itn_coverage_pct=(ITN_COLUMN, "mean"),
        mean_risk_index=(RISK_SCORE_COLUMN, "mean"),
    )
    summary.insert(
        summary.columns.get_loc("parasitemia_person_pct"),
        "weighted_parasitemia_pct",
        summary.pop("parasitemia_person_pct") / summary["population_at_risk"],
    )

    summary = summary.reset_index()
    summary["population_at_risk"] = summary["population_at_risk"].astype("int64")
    return summary.sort_values("mean_risk_index", ascending=False).reset_index(drop=True)


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


def simulate_intervention_scenario(
    df: pd.DataFrame,
    target_strata: list[str],
    coverage_increase_pct: float,
) -> pd.DataFrame:
    """Simulate a targeted ITN distribution campaign and measure counterfactual risk reduction.

    Applies a percentage-point coverage increase to selected endemicity strata,
    caps coverage at operational feasibility (98%), re-evaluates the composite risk score,
    and returns metrics on net requirements and risk reduction.

    Args:
        df: Scored national cohort frame from :func:`calculate_composite_risk_score`.
        target_strata: Endemicity zones targeted for commodity distribution.
        coverage_increase_pct: Absolute percentage-point increase in ITN coverage.

    Returns:
        DataFrame with simulated ITN coverage, counterfactual risk score, risk delta,
        and required net distribution volumes.

    Raises:
        KeyError: If a required column, including a component index, is absent.
        ValueError: If ``coverage_increase_pct`` is negative.
    """
    _require_columns(
        df,
        (
            ZONE_COLUMN,
            ITN_COLUMN,
            POPULATION_COLUMN,
            PARASITEMIA_INDEX_COLUMN,
            INCIDENCE_INDEX_COLUMN,
            NET_GAP_INDEX_COLUMN,
            RISK_SCORE_COLUMN,
        ),
    )

    if coverage_increase_pct < 0:
        raise ValueError(
            f"coverage_increase_pct must be non-negative; received {coverage_increase_pct}"
        )
    unknown_strata = [zone for zone in target_strata if zone not in ZONE_SEVERITY_ORDER]
    if unknown_strata:
        LOGGER.warning(
            "Ignoring unrecognised target stratum/strata: %s", ", ".join(unknown_strata)
        )

    simulated = df.copy()
    is_targeted = simulated[ZONE_COLUMN].isin(target_strata)

    # Apply coverage intervention bounded by operational feasibility. The cap
    # never lowers a county that already sits above it.
    original_coverage = simulated[ITN_COLUMN]
    boosted = (original_coverage + coverage_increase_pct).clip(upper=MAX_FEASIBLE_ITN_COVERAGE)
    simulated["simulated_itn_coverage"] = original_coverage.where(
        ~is_targeted, np.maximum(original_coverage, boosted)
    )

    # Rescale against the baseline bounds so untargeted counties are unaffected.
    # Left unclipped: a county pushed past the national best still earns credit,
    # and the composite below is clipped to 0-100 regardless.
    gap_min, gap_spread = _baseline_gap_bounds(simulated)
    simulated_gap = 100.0 - simulated["simulated_itn_coverage"]
    if gap_spread > 0.0:
        simulated["simulated_net_gap_index"] = (simulated_gap - gap_min) / gap_spread * 100.0
    else:
        simulated["simulated_net_gap_index"] = 0.0

    # Re-score composite risk holding burden terms constant
    simulated["simulated_risk_score"] = (
        simulated[PARASITEMIA_INDEX_COLUMN] * RISK_WEIGHTS[PARASITEMIA_INDEX_COLUMN]
        + simulated[INCIDENCE_INDEX_COLUMN] * RISK_WEIGHTS[INCIDENCE_INDEX_COLUMN]
        + simulated["simulated_net_gap_index"] * RISK_WEIGHTS[NET_GAP_INDEX_COLUMN]
    ).clip(lower=0.0, upper=100.0)

    simulated["simulated_risk_tier"] = _assign_risk_tier(simulated["simulated_risk_score"])
    simulated["risk_reduction"] = (
        simulated[RISK_SCORE_COLUMN] - simulated["simulated_risk_score"]
    ).round(2)

    # Calculate commodity allocation logistics
    actual_coverage_gain = simulated["simulated_itn_coverage"] - original_coverage
    additional_protected_pop = (simulated[POPULATION_COLUMN] * (actual_coverage_gain / 100.0))
    simulated["required_itn_commodities"] = (
        np.ceil(additional_protected_pop / PERSONS_PER_ITN).astype("int64")
    )

    return simulated


def _baseline_gap_bounds(df: pd.DataFrame) -> tuple[float, float]:
    """Return the net-gap ``(minimum, spread)`` the baseline index was scaled on.

    ``df`` may be a filtered view of a nationally scored cohort, so its own
    min/max need not match the national ones. Because the net-gap index is a
    linear map of the gap, the national bounds are recovered from the stored
    index instead, falling back to the frame's own bounds when it holds too
    little spread to invert the map.
    """
    gap = 100.0 - df[ITN_COLUMN]
    index = df[NET_GAP_INDEX_COLUMN]
    index_spread = float(index.max() - index.min())
    if index_spread > 0.0:
        gap_spread = float(gap.max() - gap.min()) * 100.0 / index_spread
        gap_min = float(gap.loc[index.idxmin()]) - float(index.min()) * gap_spread / 100.0
        return gap_min, gap_spread
    return float(gap.min()), float(gap.max() - gap.min())
