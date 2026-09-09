"""AfyaTrack: subnational malaria surveillance and analytical modeling.

The package is layered so each stage can be tested in isolation:

``src.ingestion``
    Reads extracts and enforces the schema contract.
``src.analytics``
    Derives the composite risk index, intervention correlations, and strata
    aggregates from a validated frame.
``src.visualization``
    Turns analytical frames into Plotly figures. Holds no analytical logic.
"""

__version__ = "1.0.0"
