"""Tests for :mod:`src.analytics`.

The composite index is a decision instrument, so the assertions here are about
mathematical invariants rather than remembered outputs: bounds that hold for any
input, a composite that is exactly its weighted parts, an ordering that respects
dominance, and aggregates whose totals reconcile with the rows they came from.
Pinning invariants rather than values means the tests keep their meaning if the
weights are ever retuned.
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from src.analytics import (
    INCIDENCE_INDEX_COLUMN,
    MAX_FEASIBLE_ITN_COVERAGE,
    MIN_CORRELATION_SAMPLE,
    NET_GAP_INDEX_COLUMN,
    PARASITEMIA_INDEX_COLUMN,
    RISK_SCORE_COLUMN,
    RISK_TIER_BOUNDS,
    RISK_TIER_COLUMN,
    RISK_TIER_LABELS,
    RISK_WEIGHTS,
    ZONE_SEVERITY_ORDER,
    calculate_composite_risk_score,
    evaluate_intervention_correlation,
    get_strata_summary,
    simulate_intervention_scenario,
)
from src.ingestion import ENDEMICITY_ZONES, load_surveillance_data

COMPONENT_COLUMNS = (
    PARASITEMIA_INDEX_COLUMN,
    INCIDENCE_INDEX_COLUMN,
    NET_GAP_INDEX_COLUMN,
)

STRATA_SUMMARY_COLUMNS = (
    "endemicity_zone",
    "county_count",
    "population_at_risk",
    "mean_parasitemia_pct",
    "weighted_parasitemia_pct",
    "mean_itn_coverage_pct",
    "mean_risk_index",
)


@pytest.fixture(scope="module")
def national_cohort(reference_path) -> pd.DataFrame:
    """The validated 47-county extract, as the dashboard consumes it."""
    return load_surveillance_data(str(reference_path))


@pytest.fixture
def graded_frame() -> pd.DataFrame:
    """Three counties on a strict dominance ordering.

    ``Alpha`` is worst on all three signals and ``Gamma`` best on all three, so
    their scores are exactly 100 and 0 and the middle row is strictly between.
    """
    return pd.DataFrame(
        [
            {
                "county_name": "Alpha",
                "endemicity_zone": "Lake Endemic",
                "population": 1_000_000,
                "parasitemia_rate_rdt_pct": 30.0,
                "itn_coverage_pct": 40.0,
                "annual_rainfall_mm": 1500,
                "confirmed_cases_per_1000": 400.0,
            },
            {
                "county_name": "Beta",
                "endemicity_zone": "Coast Endemic",
                "population": 500_000,
                "parasitemia_rate_rdt_pct": 15.0,
                "itn_coverage_pct": 60.0,
                "annual_rainfall_mm": 1000,
                "confirmed_cases_per_1000": 200.0,
            },
            {
                "county_name": "Gamma",
                "endemicity_zone": "Low Risk",
                "population": 250_000,
                "parasitemia_rate_rdt_pct": 1.0,
                "itn_coverage_pct": 90.0,
                "annual_rainfall_mm": 800,
                "confirmed_cases_per_1000": 5.0,
            },
        ]
    )


class TestRiskModelDefinition:
    """The weighting scheme itself must stay coherent."""

    def test_severity_order_matches_the_ingestion_contract(self) -> None:
        assert ZONE_SEVERITY_ORDER == ENDEMICITY_ZONES

    def test_weights_sum_to_one(self) -> None:
        assert sum(RISK_WEIGHTS.values()) == pytest.approx(1.0)

    def test_weights_are_declared_for_every_component(self) -> None:
        assert tuple(RISK_WEIGHTS) == COMPONENT_COLUMNS

    def test_tier_bounds_and_labels_are_consistent(self) -> None:
        assert len(RISK_TIER_BOUNDS) == len(RISK_TIER_LABELS) + 1
        assert list(RISK_TIER_BOUNDS) == sorted(RISK_TIER_BOUNDS)


class TestCompositeRiskBounds:
    """The index is bounded to 0-100 by construction; prove it holds."""

    def test_national_scores_stay_within_bounds(self, national_cohort: pd.DataFrame) -> None:
        scored = calculate_composite_risk_score(national_cohort)
        assert scored[RISK_SCORE_COLUMN].between(0.0, 100.0).all()

    def test_components_stay_within_bounds(self, national_cohort: pd.DataFrame) -> None:
        scored = calculate_composite_risk_score(national_cohort)
        for component in COMPONENT_COLUMNS:
            assert scored[component].between(0.0, 100.0).all(), component

    def test_bounds_hold_under_extreme_inputs(self) -> None:
        extremes = pd.DataFrame(
            {
                "county_name": ["Floor", "Ceiling", "Middle"],
                "endemicity_zone": ["Low Risk", "Lake Endemic", "Coast Endemic"],
                "population": [1_000, 25_000_000, 900_000],
                "parasitemia_rate_rdt_pct": [0.0, 100.0, 50.0],
                "itn_coverage_pct": [100.0, 0.0, 50.0],
                "annual_rainfall_mm": [0.0, 4_000.0, 900.0],
                "confirmed_cases_per_1000": [0.0, 1_000.0, 400.0],
            }
        )
        scored = calculate_composite_risk_score(extremes)
        assert scored[RISK_SCORE_COLUMN].between(0.0, 100.0).all()

    def test_dominant_county_scores_100_and_dominated_scores_0(
        self, graded_frame: pd.DataFrame
    ) -> None:
        scored = calculate_composite_risk_score(graded_frame).set_index("county_name")
        assert scored.loc["Alpha", RISK_SCORE_COLUMN] == pytest.approx(100.0)
        assert scored.loc["Gamma", RISK_SCORE_COLUMN] == pytest.approx(0.0)

    def test_dominance_implies_strictly_higher_score(
        self, graded_frame: pd.DataFrame
    ) -> None:
        scored = calculate_composite_risk_score(graded_frame).set_index("county_name")
        scores = scored[RISK_SCORE_COLUMN]
        assert scores["Alpha"] > scores["Beta"] > scores["Gamma"]

    def test_degenerate_cohort_scores_zero(self) -> None:
        """A single county has no cohort to rank against, so it scores 0."""
        single = pd.DataFrame(
            [
                {
                    "county_name": "Solo",
                    "endemicity_zone": "Lake Endemic",
                    "population": 1_000_000,
                    "parasitemia_rate_rdt_pct": 22.0,
                    "itn_coverage_pct": 70.0,
                    "annual_rainfall_mm": 1400,
                    "confirmed_cases_per_1000": 300.0,
                }
            ]
        )
        scored = calculate_composite_risk_score(single)
        assert scored[RISK_SCORE_COLUMN].iloc[0] == pytest.approx(0.0)


class TestCompositeRiskComposition:
    """The composite must be exactly its declared weighted parts."""

    def test_composite_equals_weighted_component_sum(
        self, national_cohort: pd.DataFrame
    ) -> None:
        scored = calculate_composite_risk_score(national_cohort)
        recomputed = sum(
            scored[component] * weight for component, weight in RISK_WEIGHTS.items()
        )
        pd.testing.assert_series_equal(
            scored[RISK_SCORE_COLUMN], recomputed, check_names=False
        )

    def test_coverage_enters_as_a_gap_not_a_level(self, graded_frame: pd.DataFrame) -> None:
        """The county with the lowest ITN coverage carries the highest net-gap index."""
        scored = calculate_composite_risk_score(graded_frame).set_index("county_name")
        assert scored[NET_GAP_INDEX_COLUMN].idxmax() == "Alpha"
        assert scored[NET_GAP_INDEX_COLUMN].idxmin() == "Gamma"

    def test_original_columns_are_preserved(self, national_cohort: pd.DataFrame) -> None:
        scored = calculate_composite_risk_score(national_cohort)
        assert set(national_cohort.columns).issubset(set(scored.columns))

    def test_input_frame_is_not_mutated(self, national_cohort: pd.DataFrame) -> None:
        before = national_cohort.copy(deep=True)
        calculate_composite_risk_score(national_cohort)
        pd.testing.assert_frame_equal(national_cohort, before)

    def test_missing_column_raises_key_error(self, national_cohort: pd.DataFrame) -> None:
        with pytest.raises(KeyError, match="itn_coverage_pct"):
            calculate_composite_risk_score(national_cohort.drop(columns=["itn_coverage_pct"]))


class TestRiskTiers:
    """Tier assignment must follow the published cut points exactly."""

    def test_tiers_match_declared_cut_points(self, national_cohort: pd.DataFrame) -> None:
        scored = calculate_composite_risk_score(national_cohort)
        expected = pd.cut(
            scored[RISK_SCORE_COLUMN],
            bins=list(RISK_TIER_BOUNDS),
            labels=list(RISK_TIER_LABELS),
            include_lowest=True,
        )
        assert (scored[RISK_TIER_COLUMN].astype(str) == expected.astype(str)).all()

    def test_tier_is_an_ordered_categorical(self, national_cohort: pd.DataFrame) -> None:
        scored = calculate_composite_risk_score(national_cohort)
        dtype = scored[RISK_TIER_COLUMN].dtype
        assert isinstance(dtype, pd.CategoricalDtype)
        assert dtype.ordered
        assert list(dtype.categories) == list(RISK_TIER_LABELS)

    def test_every_county_receives_a_tier(self, national_cohort: pd.DataFrame) -> None:
        scored = calculate_composite_risk_score(national_cohort)
        assert scored[RISK_TIER_COLUMN].notna().all()


class TestInterventionCorrelation:
    """The OLS fit and coefficient must be recoverable and guarded."""

    def test_reports_the_documented_keys(self, national_cohort: pd.DataFrame) -> None:
        result = evaluate_intervention_correlation(national_cohort)
        assert set(result) == {
            "pearson_r",
            "r_squared",
            "slope",
            "intercept",
            "t_statistic",
            "n_counties",
            "direction",
        }
        assert result["n_counties"] == len(national_cohort)

    def test_coefficient_stays_within_unit_interval(
        self, national_cohort: pd.DataFrame
    ) -> None:
        result = evaluate_intervention_correlation(national_cohort)
        assert -1.0 <= float(result["pearson_r"]) <= 1.0
        assert 0.0 <= float(result["r_squared"]) <= 1.0

    def test_recovers_a_known_positive_line(self) -> None:
        """parasitemia = 2 * coverage + 1 must return r = 1, slope 2, intercept 1."""
        coverage = [10.0, 20.0, 30.0, 40.0]
        frame = pd.DataFrame(
            {
                "itn_coverage_pct": coverage,
                "parasitemia_rate_rdt_pct": [2.0 * x + 1.0 for x in coverage],
            }
        )
        result = evaluate_intervention_correlation(frame)
        assert float(result["pearson_r"]) == pytest.approx(1.0)
        assert float(result["slope"]) == pytest.approx(2.0)
        assert float(result["intercept"]) == pytest.approx(1.0)

    def test_recovers_a_known_inverse_line(self) -> None:
        coverage = [40.0, 55.0, 70.0, 85.0]
        frame = pd.DataFrame(
            {
                "itn_coverage_pct": coverage,
                "parasitemia_rate_rdt_pct": [60.0 - 0.5 * x for x in coverage],
            }
        )
        result = evaluate_intervention_correlation(frame)
        assert float(result["pearson_r"]) == pytest.approx(-1.0)
        assert float(result["slope"]) == pytest.approx(-0.5)
        assert "inverse" in str(result["direction"])

    def test_perfect_inverse_fit_has_negative_t_statistic(self) -> None:
        frame = pd.DataFrame(
            {
                "itn_coverage_pct": [40.0, 55.0, 70.0],
                "parasitemia_rate_rdt_pct": [30.0, 20.0, 10.0],
            }
        )
        result = evaluate_intervention_correlation(frame)
        assert float(result["t_statistic"]) == float("-inf")

    def test_rejects_a_sample_too_small_to_fit(self) -> None:
        frame = pd.DataFrame(
            {
                "itn_coverage_pct": [40.0, 60.0],
                "parasitemia_rate_rdt_pct": [10.0, 5.0],
            }
        )
        with pytest.raises(ValueError, match=str(MIN_CORRELATION_SAMPLE)):
            evaluate_intervention_correlation(frame)

    def test_rejects_zero_variance_input(self) -> None:
        frame = pd.DataFrame(
            {
                "itn_coverage_pct": [70.0, 70.0, 70.0],
                "parasitemia_rate_rdt_pct": [10.0, 5.0, 8.0],
            }
        )
        with pytest.raises(ValueError, match="zero variance"):
            evaluate_intervention_correlation(frame)

    def test_missing_column_raises_key_error(self) -> None:
        frame = pd.DataFrame({"itn_coverage_pct": [40.0, 50.0, 60.0]})
        with pytest.raises(KeyError, match="parasitemia_rate_rdt_pct"):
            evaluate_intervention_correlation(frame)


class TestStrataSummary:
    """Stratum aggregates drive commodity allocation, so totals must reconcile."""

    def test_output_schema_is_stable(self, national_cohort: pd.DataFrame) -> None:
        summary = get_strata_summary(national_cohort)
        assert tuple(summary.columns) == STRATA_SUMMARY_COLUMNS

    def test_one_row_per_stratum_present(self, national_cohort: pd.DataFrame) -> None:
        summary = get_strata_summary(national_cohort)
        assert len(summary) == national_cohort["endemicity_zone"].nunique()
        assert set(summary["endemicity_zone"]) <= set(ZONE_SEVERITY_ORDER)

    def test_county_counts_reconcile_with_the_cohort(
        self, national_cohort: pd.DataFrame
    ) -> None:
        summary = get_strata_summary(national_cohort)
        assert int(summary["county_count"].sum()) == len(national_cohort)

    def test_population_totals_reconcile_with_the_cohort(
        self, national_cohort: pd.DataFrame
    ) -> None:
        summary = get_strata_summary(national_cohort)
        assert int(summary["population_at_risk"].sum()) == int(
            national_cohort["population"].sum()
        )

    def test_mean_parasitemia_matches_a_direct_group_mean(
        self, national_cohort: pd.DataFrame
    ) -> None:
        summary = get_strata_summary(national_cohort).set_index("endemicity_zone")
        expected = national_cohort.groupby("endemicity_zone")["parasitemia_rate_rdt_pct"].mean()
        for zone, mean_value in expected.items():
            assert summary.loc[zone, "mean_parasitemia_pct"] == pytest.approx(mean_value)

    def test_weighted_parasitemia_matches_a_direct_weighted_mean(
        self, national_cohort: pd.DataFrame
    ) -> None:
        summary = get_strata_summary(national_cohort).set_index("endemicity_zone")
        for zone, group in national_cohort.groupby("endemicity_zone"):
            expected = (
                group["parasitemia_rate_rdt_pct"] * group["population"]
            ).sum() / group["population"].sum()
            assert summary.loc[zone, "weighted_parasitemia_pct"] == pytest.approx(expected)

    def test_rows_are_sorted_by_risk_descending(self, national_cohort: pd.DataFrame) -> None:
        summary = get_strata_summary(national_cohort)
        assert summary["mean_risk_index"].is_monotonic_decreasing

    def test_lake_endemic_leads_the_national_ranking(
        self, national_cohort: pd.DataFrame
    ) -> None:
        """A sanity check on the seed data's epidemiology, not just the code."""
        summary = get_strata_summary(national_cohort)
        assert summary.iloc[0]["endemicity_zone"] == "Lake Endemic"

    def test_scores_the_frame_when_no_index_is_present(
        self, national_cohort: pd.DataFrame
    ) -> None:
        assert RISK_SCORE_COLUMN not in national_cohort.columns
        summary = get_strata_summary(national_cohort)
        assert summary["mean_risk_index"].between(0.0, 100.0).all()

    def test_reuses_an_existing_index_without_rescoring(
        self, national_cohort: pd.DataFrame
    ) -> None:
        scored = calculate_composite_risk_score(national_cohort)
        from_scored = get_strata_summary(scored)
        from_raw = get_strata_summary(national_cohort)
        pd.testing.assert_frame_equal(from_scored, from_raw)

    def test_population_totals_are_integral(self, national_cohort: pd.DataFrame) -> None:
        summary = get_strata_summary(national_cohort)
        assert pd.api.types.is_integer_dtype(summary["population_at_risk"])

    def test_missing_column_raises_key_error(self, national_cohort: pd.DataFrame) -> None:
        with pytest.raises(KeyError, match="population"):
            get_strata_summary(national_cohort.drop(columns=["population"]))


class TestScenarioSimulation:
    """Validate counterfactual intervention simulation invariants."""

    def test_untargeted_strata_remain_unmodified(self, national_cohort: pd.DataFrame) -> None:
        scored = calculate_composite_risk_score(national_cohort)
        simulated = simulate_intervention_scenario(
            scored, target_strata=["Lake Endemic"], coverage_increase_pct=15.0
        )
        non_lake = simulated[simulated["endemicity_zone"] != "Lake Endemic"]
        assert (non_lake["risk_reduction"] == 0.0).all()
        assert (non_lake["required_itn_commodities"] == 0).all()

    def test_coverage_caps_at_feasibility_limit(self, national_cohort: pd.DataFrame) -> None:
        scored = calculate_composite_risk_score(national_cohort)
        simulated = simulate_intervention_scenario(
            scored, target_strata=["Lake Endemic"], coverage_increase_pct=50.0
        )
        assert (simulated["simulated_itn_coverage"] <= MAX_FEASIBLE_ITN_COVERAGE).all()

    def test_risk_reduction_is_non_negative(self, national_cohort: pd.DataFrame) -> None:
        scored = calculate_composite_risk_score(national_cohort)
        simulated = simulate_intervention_scenario(
            scored, target_strata=["Lake Endemic", "Coast Endemic"], coverage_increase_pct=10.0
        )
        assert (simulated["risk_reduction"] >= 0.0).all()

    def test_negative_coverage_increase_is_rejected(self, national_cohort: pd.DataFrame) -> None:
        scored = calculate_composite_risk_score(national_cohort)
        with pytest.raises(ValueError, match="non-negative"):
            simulate_intervention_scenario(
                scored, target_strata=["Lake Endemic"], coverage_increase_pct=-5.0
            )

    def test_unscored_frame_raises_key_error(self, national_cohort: pd.DataFrame) -> None:
        partially_scored = national_cohort.assign(composite_risk_score=50.0)
        with pytest.raises(KeyError, match="parasitemia_index"):
            simulate_intervention_scenario(
                partially_scored, target_strata=["Lake Endemic"], coverage_increase_pct=10.0
            )

    def test_unknown_stratum_is_logged(
        self, national_cohort: pd.DataFrame, caplog: pytest.LogCaptureFixture
    ) -> None:
        scored = calculate_composite_risk_score(national_cohort)
        with caplog.at_level(logging.WARNING, logger="src.analytics"):
            simulated = simulate_intervention_scenario(
                scored, target_strata=["Lake endemic"], coverage_increase_pct=10.0
            )
        assert "Lake endemic" in caplog.text
        assert (simulated["required_itn_commodities"] == 0).all()

    def test_coverage_above_cap_is_never_lowered(self, national_cohort: pd.DataFrame) -> None:
        saturated = national_cohort.copy()
        saturated.loc[0, "itn_coverage_pct"] = 99.0
        scored = calculate_composite_risk_score(saturated)
        simulated = simulate_intervention_scenario(
            scored, target_strata=[scored.loc[0, "endemicity_zone"]], coverage_increase_pct=5.0
        )
        assert simulated.loc[0, "simulated_itn_coverage"] == pytest.approx(99.0)
        assert (simulated["required_itn_commodities"] >= 0).all()

    def test_filtered_view_leaves_untargeted_counties_unchanged(
        self, national_cohort: pd.DataFrame
    ) -> None:
        """A sidebar-filtered subset must still be simulated on national bounds."""
        scored = calculate_composite_risk_score(national_cohort)
        subset = scored[scored["population"].between(500_000, 1_500_000)]
        simulated = simulate_intervention_scenario(
            subset, target_strata=["Lake Endemic"], coverage_increase_pct=15.0
        )
        untargeted = simulated[simulated["endemicity_zone"] != "Lake Endemic"]
        assert (untargeted["risk_reduction"] == 0.0).all()
