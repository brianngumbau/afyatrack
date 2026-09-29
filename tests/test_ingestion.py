"""Contract tests for :mod:`src.ingestion`.

These tests pin the boundary between "the export is broken, stop" and "this row
is broken, quarantine it". That split is the module's core design decision, so
each half is asserted explicitly rather than inferred from a happy-path load.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

from src.ingestion import (
    ENDEMICITY_ZONES,
    NUMERIC_FIELDS,
    REQUIRED_COLUMNS,
    SchemaError,
    load_surveillance_data,
    validate_surveillance_frame,
)

EXPECTED_COUNTY_COUNT = 47


class TestReferenceExtract:
    """Guard the committed seed dataset against silent corruption."""

    def test_loads_every_county(self, reference_path: Path) -> None:
        frame = load_surveillance_data(str(reference_path))
        assert len(frame) == EXPECTED_COUNTY_COUNT

    def test_exposes_exactly_the_contract_columns(self, reference_path: Path) -> None:
        frame = load_surveillance_data(str(reference_path))
        assert tuple(frame.columns) == REQUIRED_COLUMNS

    def test_every_numeric_column_is_numeric(self, reference_path: Path) -> None:
        frame = load_surveillance_data(str(reference_path))
        for field in NUMERIC_FIELDS:
            assert pd.api.types.is_numeric_dtype(frame[field.name]), field.name

    @pytest.mark.parametrize("field", NUMERIC_FIELDS, ids=lambda f: f.name)
    def test_values_sit_inside_declared_bounds(self, reference_path: Path, field) -> None:
        frame = load_surveillance_data(str(reference_path))
        assert not field.violations(frame[field.name]).any()

    def test_all_five_strata_are_represented(self, reference_path: Path) -> None:
        frame = load_surveillance_data(str(reference_path))
        assert set(frame["endemicity_zone"]) == set(ENDEMICITY_ZONES)

    def test_county_names_are_unique(self, reference_path: Path) -> None:
        frame = load_surveillance_data(str(reference_path))
        assert not frame["county_name"].duplicated().any()

    def test_index_is_contiguous_after_load(self, reference_path: Path) -> None:
        frame = load_surveillance_data(str(reference_path))
        assert frame.index.tolist() == list(range(EXPECTED_COUNTY_COUNT))


class TestStructuralRejection:
    """Defects that no row filtering can repair must abort ingestion."""

    @pytest.mark.parametrize("dropped_column", REQUIRED_COLUMNS)
    def test_missing_required_column_raises(
        self, valid_frame: pd.DataFrame, dropped_column: str
    ) -> None:
        incomplete = valid_frame.drop(columns=[dropped_column])
        with pytest.raises(SchemaError, match=dropped_column):
            validate_surveillance_frame(incomplete)

    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_surveillance_data(str(tmp_path / "absent.csv"))

    def test_directory_path_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_surveillance_data(str(tmp_path))

    def test_non_utf8_file_raises_schema_error(
        self, valid_records: list[dict[str, object]], tmp_path: Path
    ) -> None:
        valid_records[0]["county_name"] = "Busiá"
        latin = tmp_path / "latin.csv"
        pd.DataFrame(valid_records).to_csv(latin, index=False, encoding="cp1252")
        with pytest.raises(SchemaError, match="UTF-8"):
            load_surveillance_data(str(latin))

    def test_empty_file_raises_schema_error(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.csv"
        empty.write_text("", encoding="utf-8")
        with pytest.raises(SchemaError):
            load_surveillance_data(str(empty))

    def test_extract_with_no_valid_rows_raises(self, valid_frame: pd.DataFrame) -> None:
        unusable = valid_frame.assign(endemicity_zone="Temperate Highlands")
        with pytest.raises(SchemaError, match="No rows survived"):
            validate_surveillance_frame(unusable)


class TestRowQuarantine:
    """Row-level defects drop one county and leave the rest intact."""

    def test_valid_frame_passes_through_unchanged(self, valid_frame: pd.DataFrame) -> None:
        validated = validate_surveillance_frame(valid_frame)
        assert len(validated) == len(valid_frame)
        assert validated["county_name"].tolist() == valid_frame["county_name"].tolist()

    def test_undeclared_columns_are_discarded(self, valid_frame: pd.DataFrame) -> None:
        noisy = valid_frame.assign(reporting_officer="ignored", extract_id=7)
        validated = validate_surveillance_frame(noisy)
        assert tuple(validated.columns) == REQUIRED_COLUMNS

    def test_parasitemia_above_100_is_quarantined(
        self, valid_records: list[dict[str, object]]
    ) -> None:
        valid_records[1]["parasitemia_rate_rdt_pct"] = 140.0
        validated = validate_surveillance_frame(pd.DataFrame(valid_records))
        assert "Kilifi" not in set(validated["county_name"])
        assert len(validated) == len(valid_records) - 1

    def test_negative_rate_is_quarantined(
        self, valid_records: list[dict[str, object]]
    ) -> None:
        valid_records[2]["parasitemia_rate_rdt_pct"] = -3.1
        validated = validate_surveillance_frame(pd.DataFrame(valid_records))
        assert "Turkana" not in set(validated["county_name"])

    @pytest.mark.parametrize("boundary", [0.0, 100.0])
    def test_inclusive_bounds_are_retained(
        self, valid_records: list[dict[str, object]], boundary: float
    ) -> None:
        valid_records[1]["parasitemia_rate_rdt_pct"] = boundary
        valid_records[1]["itn_coverage_pct"] = boundary
        validated = validate_surveillance_frame(pd.DataFrame(valid_records))
        assert "Kilifi" in set(validated["county_name"])

    def test_non_numeric_cell_is_coerced_then_quarantined(
        self, valid_records: list[dict[str, object]]
    ) -> None:
        valid_records[0]["population"] = "not available"
        validated = validate_surveillance_frame(pd.DataFrame(valid_records))
        assert "Siaya" not in set(validated["county_name"])
        assert pd.api.types.is_numeric_dtype(validated["population"])

    def test_missing_cell_is_quarantined(
        self, valid_records: list[dict[str, object]]
    ) -> None:
        valid_records[3]["annual_rainfall_mm"] = None
        validated = validate_surveillance_frame(pd.DataFrame(valid_records))
        assert "Nyeri" not in set(validated["county_name"])

    def test_blank_county_name_is_quarantined(
        self, valid_records: list[dict[str, object]]
    ) -> None:
        valid_records[0]["county_name"] = "   "
        validated = validate_surveillance_frame(pd.DataFrame(valid_records))
        assert len(validated) == len(valid_records) - 1

    def test_unrecognised_zone_is_quarantined(
        self, valid_records: list[dict[str, object]]
    ) -> None:
        valid_records[2]["endemicity_zone"] = "Arid Fringe"
        validated = validate_surveillance_frame(pd.DataFrame(valid_records))
        assert set(validated["endemicity_zone"]).issubset(set(ENDEMICITY_ZONES))
        assert "Turkana" not in set(validated["county_name"])

    def test_surrounding_whitespace_is_normalised(
        self, valid_records: list[dict[str, object]]
    ) -> None:
        valid_records[0]["county_name"] = "  Siaya  "
        valid_records[0]["endemicity_zone"] = " Lake   Endemic "
        validated = validate_surveillance_frame(pd.DataFrame(valid_records))
        row = validated.loc[validated["county_name"] == "Siaya"]
        assert len(row) == 1
        assert row.iloc[0]["endemicity_zone"] == "Lake Endemic"

    def test_duplicate_county_keeps_first_record(
        self, valid_records: list[dict[str, object]]
    ) -> None:
        duplicate = dict(valid_records[0], parasitemia_rate_rdt_pct=11.1)
        validated = validate_surveillance_frame(pd.DataFrame([*valid_records, duplicate]))
        siaya = validated.loc[validated["county_name"] == "Siaya"]
        assert len(siaya) == 1
        assert siaya.iloc[0]["parasitemia_rate_rdt_pct"] == pytest.approx(26.9)

    def test_duplicate_county_is_matched_case_insensitively(
        self, valid_records: list[dict[str, object]]
    ) -> None:
        duplicate = dict(valid_records[0], county_name="SIAYA")
        validated = validate_surveillance_frame(pd.DataFrame([*valid_records, duplicate]))
        assert len(validated) == len(valid_records)
        assert "SIAYA" not in set(validated["county_name"])

    def test_fractional_population_is_quarantined(
        self, valid_records: list[dict[str, object]], caplog: pytest.LogCaptureFixture
    ) -> None:
        valid_records[2]["population"] = 1_034_000.5
        with caplog.at_level(logging.WARNING, logger="src.ingestion"):
            validated = validate_surveillance_frame(pd.DataFrame(valid_records))
        assert "Turkana" not in set(validated["county_name"])
        assert "whole number" in caplog.text

    def test_population_is_integer_typed(self, valid_frame: pd.DataFrame) -> None:
        validated = validate_surveillance_frame(valid_frame.astype({"population": float}))
        assert pd.api.types.is_integer_dtype(validated["population"])

    def test_quarantined_rows_are_logged_by_name(
        self, valid_records: list[dict[str, object]], caplog: pytest.LogCaptureFixture
    ) -> None:
        valid_records[1]["itn_coverage_pct"] = 250.0
        with caplog.at_level(logging.WARNING, logger="src.ingestion"):
            validate_surveillance_frame(pd.DataFrame(valid_records))
        assert "Kilifi" in caplog.text
        assert "itn_coverage_pct" in caplog.text


class TestFileRoundTrip:
    """The disk path and the in-memory path must agree."""

    def test_load_matches_direct_validation(
        self, valid_records: list[dict[str, object]], write_csv
    ) -> None:
        path = write_csv(valid_records)
        from_disk = load_surveillance_data(str(path))
        in_memory = validate_surveillance_frame(pd.DataFrame(valid_records))
        pd.testing.assert_frame_equal(from_disk, in_memory, check_dtype=False)
