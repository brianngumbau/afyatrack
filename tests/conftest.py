"""Shared fixtures for the AfyaTrack test suite.

Two families of fixture are provided deliberately. Synthetic frames are small
and hand-computable, so assertions about them can state exact expected values
rather than tolerances. The reference fixture loads the committed 47-county
extract, which guards the seed data itself against regression -- a malformed
edit to the CSV should fail CI, not surface as a blank dashboard.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_EXTRACT = PROJECT_ROOT / "data" / "kenya_malaria_surveillance.csv"

#: Four counties spanning the transmission gradient. Values are chosen so the
#: first row dominates the last on every signal, which makes the composite
#: index's ordering assertions exact rather than approximate.
BASE_RECORDS: list[dict[str, object]] = [
    {
        "county_name": "Siaya",
        "endemicity_zone": "Lake Endemic",
        "population": 1_108_000,
        "parasitemia_rate_rdt_pct": 26.9,
        "itn_coverage_pct": 60.0,
        "annual_rainfall_mm": 1385,
        "confirmed_cases_per_1000": 341.7,
    },
    {
        "county_name": "Kilifi",
        "endemicity_zone": "Coast Endemic",
        "population": 1_622_000,
        "parasitemia_rate_rdt_pct": 8.4,
        "itn_coverage_pct": 74.6,
        "annual_rainfall_mm": 1085,
        "confirmed_cases_per_1000": 118.7,
    },
    {
        "county_name": "Turkana",
        "endemicity_zone": "Semi-Arid Seasonal",
        "population": 1_034_000,
        "parasitemia_rate_rdt_pct": 2.6,
        "itn_coverage_pct": 52.4,
        "annual_rainfall_mm": 430,
        "confirmed_cases_per_1000": 34.9,
    },
    {
        "county_name": "Nyeri",
        "endemicity_zone": "Low Risk",
        "population": 847_000,
        "parasitemia_rate_rdt_pct": 0.2,
        "itn_coverage_pct": 90.0,
        "annual_rainfall_mm": 1105,
        "confirmed_cases_per_1000": 2.9,
    },
]


@pytest.fixture
def valid_records() -> list[dict[str, object]]:
    """Return a mutable copy of the baseline records for per-test tampering."""
    return [dict(record) for record in BASE_RECORDS]


@pytest.fixture
def valid_frame(valid_records: list[dict[str, object]]) -> pd.DataFrame:
    """A schema-conformant four-county frame."""
    return pd.DataFrame(valid_records)


@pytest.fixture
def write_csv(tmp_path: Path):
    """Return a factory that materialises records as a CSV and yields its path."""

    def _write(records: list[dict[str, object]], name: str = "extract.csv") -> Path:
        path = tmp_path / name
        pd.DataFrame(records).to_csv(path, index=False)
        return path

    return _write


@pytest.fixture(scope="session")
def reference_path() -> Path:
    """Filesystem path to the committed 47-county seed extract."""
    return REFERENCE_EXTRACT
