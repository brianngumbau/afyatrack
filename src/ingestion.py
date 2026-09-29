"""CSV ingestion and schema enforcement for the AfyaTrack pipeline.

This module is the only door through which raw surveillance extracts enter the
platform. Nothing downstream re-validates, so :func:`load_surveillance_data`
either returns a frame that satisfies the contract below in full, or it raises.
That guarantee is what lets :mod:`src.analytics` compute on every column without
defensive null handling.

Two classes of defect are treated differently on purpose:

* **Contract defects** -- a missing column, an unreadable file, an extract with
  no usable rows left -- abort ingestion with :class:`SchemaError`. They mean
  the upstream export itself is wrong and a human has to look at it.
* **Row defects** -- a blank cell, a parasitemia rate of 140%, an unrecognised
  endemicity zone -- drop the offending county and emit a warning naming it.
  A single malformed district must not cost us the other 46.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

LOGGER = logging.getLogger(__name__)

#: Endemicity strata used by the Kenya Malaria Strategy. Any other label is
#: treated as an upstream coding error rather than a new stratum, because the
#: risk model in :mod:`src.analytics` is calibrated against these five only.
ENDEMICITY_ZONES: Final[tuple[str, ...]] = (
    "Lake Endemic",
    "Coast Endemic",
    "Highland Epidemic",
    "Semi-Arid Seasonal",
    "Low Risk",
)

COUNTY_COLUMN: Final[str] = "county_name"
ZONE_COLUMN: Final[str] = "endemicity_zone"


class SchemaError(ValueError):
    """Raised when an extract violates the ingestion contract structurally.

    Signals a defect that no amount of row filtering can repair: absent
    columns, an empty export, or a file in which every row failed validation.
    """


@dataclass(frozen=True)
class NumericField:
    """A numeric column together with the closed interval it must fall in.

    The bounds are epidemiological plausibility limits, not observed extremes.
    They are deliberately generous -- the job here is to catch unit errors and
    corrupted exports (a rate recorded as 0-1 instead of 0-100, a population in
    thousands), not to second-guess a legitimately unusual county.
    """

    name: str
    minimum: float
    maximum: float
    unit: str
    integer: bool = False

    def violations(self, series: pd.Series) -> pd.Series:
        """Return a boolean mask of values that fall outside the interval.

        Nulls are excluded: absent values are handled by the completeness pass
        so that each row is reported against exactly one failure reason.
        """
        return series.notna() & ~series.between(self.minimum, self.maximum)

    def fractional(self, series: pd.Series) -> pd.Series:
        """Return a boolean mask of non-whole values in a count field.

        Always empty for fields not declared ``integer``. Nulls are excluded for
        the same reason as in :meth:`violations`.
        """
        if not self.integer:
            return pd.Series(False, index=series.index)
        return series.notna() & (series % 1 != 0)

    def describe_range(self) -> str:
        """Render the accepted interval for log and error messages."""
        return f"{self.minimum:g}-{self.maximum:g} {self.unit}"


#: The numeric contract. Ranges follow WHO/DHIS2 reporting conventions: rates
#: and coverage are percentages on a 0-100 scale, never proportions.
NUMERIC_FIELDS: Final[tuple[NumericField, ...]] = (
    NumericField("population", 1_000, 25_000_000, "residents", integer=True),
    NumericField("parasitemia_rate_rdt_pct", 0.0, 100.0, "%"),
    NumericField("itn_coverage_pct", 0.0, 100.0, "%"),
    NumericField("annual_rainfall_mm", 0.0, 4_000.0, "mm"),
    NumericField("confirmed_cases_per_1000", 0.0, 1_000.0, "cases/1,000"),
)

REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    (COUNTY_COLUMN, ZONE_COLUMN) + tuple(field.name for field in NUMERIC_FIELDS)
)


def load_surveillance_data(filepath: str) -> pd.DataFrame:
    """Read a surveillance extract from disk and enforce the schema contract.

    Args:
        filepath: Path to a UTF-8 CSV extract holding one row per county.

    Returns:
        A validated frame indexed 0..n-1, carrying exactly the columns in
        :data:`REQUIRED_COLUMNS` with numeric dtypes coerced and every row
        inside its declared bounds.

    Raises:
        FileNotFoundError: If ``filepath`` does not resolve to a file.
        SchemaError: If the file cannot be decoded or parsed, omits a required column, or
            retains no valid rows once row-level validation has run.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Surveillance extract not found: {path}")

    LOGGER.info("Reading surveillance extract from %s", path)
    try:
        raw = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise SchemaError(f"Extract {path} contains no parsable rows") from exc
    except pd.errors.ParserError as exc:
        raise SchemaError(f"Extract {path} is not well-formed CSV: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise SchemaError(f"Extract {path} is not UTF-8 encoded: {exc}") from exc

    frame = validate_surveillance_frame(raw)
    LOGGER.info(
        "Ingested %d counties covering %s residents",
        len(frame),
        f"{int(frame['population'].sum()):,}",
    )
    return frame


def validate_surveillance_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the full validation chain to an in-memory frame.

    Split out from :func:`load_surveillance_data` so that callers holding a
    frame from another source -- a database cursor, a test fixture, an API
    response -- get identical guarantees without staging a temporary file.

    Args:
        frame: Candidate surveillance data. Not mutated.

    Returns:
        A new validated frame, re-indexed contiguously.

    Raises:
        SchemaError: On a missing required column or an empty result set.
    """
    _assert_required_columns(frame)

    validated = frame.loc[:, list(REQUIRED_COLUMNS)].copy()
    validated = _normalise_text(validated)
    validated = _coerce_numeric(validated)
    validated = _drop_incomplete_rows(validated)
    validated = _drop_out_of_range_rows(validated)
    validated = _drop_fractional_rows(validated)
    validated = _drop_unknown_zones(validated)
    validated = _drop_duplicate_counties(validated)

    if validated.empty:
        raise SchemaError(
            "No rows survived validation; the extract is unusable. "
            "Review the warnings above for the per-row rejection reasons."
        )
    # Safe only now: every surviving count is non-null and whole.
    for field in NUMERIC_FIELDS:
        if field.integer:
            validated[field.name] = validated[field.name].astype("int64")
    return validated.reset_index(drop=True)


def _assert_required_columns(frame: pd.DataFrame) -> None:
    """Fail fast when the extract omits any column the pipeline depends on."""
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise SchemaError(
            "Extract is missing required column(s): "
            f"{', '.join(missing)}. Expected: {', '.join(REQUIRED_COLUMNS)}"
        )


def _normalise_text(frame: pd.DataFrame) -> pd.DataFrame:
    """Trim and collapse whitespace in the identifier columns.

    Exports pulled from spreadsheets routinely carry trailing spaces, which
    would otherwise split one county across two groups or defeat the zone
    membership check.
    """
    for column in (COUNTY_COLUMN, ZONE_COLUMN):
        normalised = (
            frame[column].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
        )
        # `replace` rather than a boolean mask: comparing a nullable string
        # column against "" yields NA for already-null cells, which cannot be
        # used as an indexer.
        frame[column] = normalised.replace("", pd.NA)
    return frame


def _coerce_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    """Force declared numeric columns to a numeric dtype.

    Unparsable cells become NaN here and are removed by the completeness pass,
    which keeps "not a number" and "out of range" as separate, separately
    logged failure modes.
    """
    for field in NUMERIC_FIELDS:
        original = frame[field.name]
        converted = pd.to_numeric(original, errors="coerce")
        unparsable = int((converted.isna() & original.notna()).sum())
        if unparsable:
            LOGGER.warning(
                "Column '%s' held %d non-numeric value(s); coerced to null",
                field.name,
                unparsable,
            )
        frame[field.name] = converted
    return frame


def _drop_incomplete_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with a null in any required column."""
    incomplete = frame.isna().any(axis=1)
    if incomplete.any():
        _log_rejections(frame.loc[incomplete], "incomplete record")
    return frame.loc[~incomplete]


def _drop_out_of_range_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove rows whose numeric values fall outside their declared bounds."""
    for field in NUMERIC_FIELDS:
        offending = field.violations(frame[field.name])
        if offending.any():
            _log_rejections(
                frame.loc[offending],
                f"'{field.name}' outside {field.describe_range()}",
            )
            frame = frame.loc[~offending]
    return frame


def _drop_fractional_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove rows holding a non-whole value in a count field."""
    for field in NUMERIC_FIELDS:
        offending = field.fractional(frame[field.name])
        if offending.any():
            _log_rejections(frame.loc[offending], f"'{field.name}' is not a whole number")
            frame = frame.loc[~offending]
    return frame


def _drop_unknown_zones(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove rows whose endemicity label is not a recognised stratum."""
    unknown = ~frame[ZONE_COLUMN].isin(ENDEMICITY_ZONES)
    if unknown.any():
        _log_rejections(frame.loc[unknown], "unrecognised endemicity zone")
    return frame.loc[~unknown]


def _drop_duplicate_counties(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep the first record per county, matching names case-insensitively.

    Duplicates typically mean two reporting periods were concatenated. Summing
    them would silently double a county's population, so the later rows are
    dropped and surfaced rather than merged. "Busia" and "BUSIA" are the same
    county, so names are compared casefolded.
    """
    duplicated = frame[COUNTY_COLUMN].str.casefold().duplicated(keep="first")
    if duplicated.any():
        _log_rejections(frame.loc[duplicated], "duplicate county record")
    return frame.loc[~duplicated]


def _log_rejections(rejected: pd.DataFrame, reason: str) -> None:
    """Emit one warning naming the counties dropped for a given reason."""
    counties = rejected[COUNTY_COLUMN].fillna("<unnamed>").tolist()
    LOGGER.warning(
        "Dropped %d row(s) -- %s: %s",
        len(counties),
        reason,
        ", ".join(str(county) for county in counties),
    )
