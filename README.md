# AfyaTrack — Subnational Malaria Surveillance & Analytical Modeling Platform

A containerized data pipeline and analytics dashboard for county-level malaria
micro-stratification in Kenya. AfyaTrack ingests subnational surveillance
extracts, enforces a schema contract on them, derives a weighted composite risk
index across all 47 counties, and presents the result as a prioritisation matrix
that a programme officer can act on.

```
data/*.csv → ingestion (schema contract) → analytics (risk model)
           → visualization (Plotly encodings) → app.py (Streamlit UI)
```

---

## 1. Public health analytics context

Kenya does not have one malaria epidemic; it has five, and they do not respond to
the same intervention mix. A national prevalence figure averages 27.5% RDT
positivity in Busia against 0.2% in Nyandarua and describes neither. **Micro-
stratification** — resolving burden to the smallest administrative unit that
still has usable data — is what turns a national average back into an
allocation decision, and it is the organising principle of this platform it is built against.

The platform is built around three ideas that recur in that work:

**Stratification before targeting.** Counties are grouped into the five
transmission strata used by the Kenya Malaria Strategy — Lake Endemic, Coast
Endemic, Highland Epidemic, Semi-Arid Seasonal, and Low Risk. The strata are
treated as an *ordered* scale, not a set of labels, and that ordering propagates
all the way into the chart palette (a single-hue navy ramp, light to dark)
rather than being re-invented per view.

**Burden is not the same as priority.** A county already at 87% bed-net coverage
and a county at 44% coverage with the same parasitemia are not equally good
places to send the next consignment. The composite risk index therefore carries
an explicit *actionability* term alongside its two burden terms, so the ranking
answers "where does a delivered commodity change an outcome" rather than merely
"where is transmission highest."

**Association is not effect.** The intervention tab reports a positive
correlation between ITN coverage and parasitemia. This is targeting, not
iatrogenesis: mass net campaigns are directed at endemic strata, so coverage is
highest exactly where burden is highest. The dashboard states this inline
alongside the coefficient rather than leaving the reader to infer it, because a
cross-sectional ecological correlation is the single most misread statistic in
subnational surveillance.

### Dataset

`data/kenya_malaria_surveillance.csv` — 47 rows, one per county.

| Field | Type | Range | Notes |
|---|---|---|---|
| `county_name` | string | — | All 47 counties |
| `endemicity_zone` | enum | 5 strata | Kenya Malaria Strategy classification |
| `population` | int | 161k – 4.91m | 2024 projections from the 2019 census |
| `parasitemia_rate_rdt_pct` | float | 0.2 – 27.5 | RDT positivity, percent |
| `itn_coverage_pct` | float | 40.0 – 86.9 | Household bed-net usage, percent |
| `annual_rainfall_mm` | int | 405 – 1800 | Mean annual precipitation |
| `confirmed_cases_per_1000` | float | 2.6 – 352.8 | Confirmed incidence |

Values follow the epidemiological gradient of the real geography: the Lake basin
counties carry high rainfall, high parasitemia, and correspondingly high campaign
coverage; the arid north-east carries low rainfall, seasonal transmission, and
the weakest net coverage; the central highlands sit near the floor on every
transmission measure. This is a realistic seed extract for demonstrating the
pipeline, **not an official data release** — see [Limitations](#8-limitations).

---

## 2. Technical architecture

The dependency arrow runs one way, and each layer is testable without the one
above it.

| Layer | Module | Responsibility |
|---|---|---|
| Ingestion | `src/ingestion.py` | Schema contract, numeric coercion, row quarantine |
| Normalization & modeling | `src/analytics.py` | Min-max rescaling, composite index, OLS fit, strata aggregation |
| Visualization | `src/visualization.py` | Plotly encodings only — no analytical logic |
| Interface | `app.py` | Layout, filter state, formatting |

**Ingestion** separates two failure classes on purpose. *Contract defects* — a
missing column, an unparsable file, an extract with no usable rows — raise
`SchemaError` and stop the pipeline, because they mean the upstream export is
wrong and a human has to look at it. *Row defects* — a blank cell, a parasitemia
rate above 100%, an unrecognised stratum, a duplicated county — quarantine that
county and emit a `logging` warning naming it, so one malformed district never
costs the other 46. Numeric bounds are plausibility limits, not observed
extremes: they exist to catch a rate exported on a 0–1 scale, not to second-guess
an unusual county.

**Analytics** never reads from disk and never imports Streamlit, so the entire
analytical surface is unit-testable without a browser or a fixture file.

**Visualization** delegates the regression overlay on the coverage scatter back
to `evaluate_intervention_correlation`, so the line drawn on screen and the
coefficient printed beside it are the same model and cannot drift apart.

### The composite risk index

Each signal is min-max rescaled to 0–100 across the national cohort, then
combined under fixed weights:

```
norm(x) = (x - min(x)) / (max(x) - min(x)) × 100

Risk = 0.50 × norm(parasitemia_rate_rdt_pct)      # burden, survey-measured
     + 0.30 × norm(confirmed_cases_per_1000)      # burden, facility-reported
     + 0.20 × norm(100 - itn_coverage_pct)        # actionability: unmet need
```

Because the weights sum to 1 and each component is bounded to 0–100, the
composite is bounded to 0–100 **by construction** — an invariant asserted
directly in `tests/test_analytics.py` rather than assumed.

*Why these weights.* Parasitemia carries the plurality because RDT prevalence is
the least reporting-dependent of the three signals. Confirmed incidence is
weighted lower precisely because it tracks health-facility access as much as it
tracks transmission — a county with weak reporting looks healthier than it is.
The net gap enters at 20% as an actionability term rather than a burden term.

*Scoring order.* The index is computed **once over all 47 counties**, and sidebar
filters are applied to the already-scored frame. Scoring after filtering would
rescale the index against whatever subset the user selected, so a county's risk
would change as they moved a slider. Ranks stay national; the view narrows.

---

## 3. Local setup

### Docker (recommended)

```bash
docker compose up --build
```

Then open <http://localhost:8501>.

Source directories are bind-mounted read-only, so edits to `app.py` or `src/`
reload live without a rebuild; only a dependency change requires one. The image
runs as the unprivileged `afyatrack` user (uid 1001) and exposes a container
healthcheck against Streamlit's `/_stcore/health` endpoint.

```bash
docker compose down --remove-orphans      # stop and clean up
docker compose run --rm afyatrack pytest  # run the suite inside the image
```

### Native Python

Requires Python 3.11 or newer.

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
make install
make run
```

### Make targets

| Target | Action |
|---|---|
| `make install` | Install dependencies from `requirements.txt` |
| `make test` | Run the suite with `pytest --verbose` |
| `make lint` | Run `flake8` over `src`, `tests`, and `app.py` |
| `make run` | Serve the dashboard on port 8501 (`PORT=` to override) |
| `make docker-up` | Build and start the containerized stack |
| `make docker-down` | Stop the stack |
| `make docker-test` | Run the suite inside the runtime image |

---

## 4. Project structure

```
afyatrack/
├── app.py                       # Streamlit UI: layout, filters, formatting
├── src/
│   ├── ingestion.py             # Schema contract and row quarantine
│   ├── analytics.py             # Risk index, OLS fit, strata aggregation
│   └── visualization.py         # Plotly figure builders
├── tests/
│   ├── conftest.py              # Synthetic and reference fixtures
│   ├── test_ingestion.py        # Contract boundary tests
│   └── test_analytics.py        # Mathematical invariant tests
├── data/
│   └── kenya_malaria_surveillance.csv
├── .github/workflows/ci.yml     # Lint, test matrix, container validation
├── .streamlit/config.toml       # Clinical theme tokens
├── Dockerfile                   # Multi-stage, non-root runtime
├── docker-compose.yml           # Local dev stack with live reload
├── Makefile                     # Canonical developer entry points
└── setup.cfg                    # flake8 and pytest configuration
```

---

## 5. Testing

```bash
make test
```

The suite is organised around what each layer promises rather than around line
coverage.

**`tests/test_ingestion.py`** pins the contract boundary: that the committed seed
extract loads all 47 counties within its declared bounds; that dropping *any*
required column raises `SchemaError` (parametrized across all seven); that a
missing file, a directory, and an empty file each fail distinctly; and that
out-of-range values, non-numeric cells, blank names, unknown strata, and
duplicate counties are quarantined individually while the remaining rows survive
— including an assertion that the rejected county is named in the log output.

**`tests/test_analytics.py`** pins mathematical invariants rather than remembered
outputs, so the tests keep their meaning if the weights are ever retuned: that
the weights sum to 1; that the index stays within 0–100 on the real cohort, under
extreme synthetic inputs, and on a degenerate single-row cohort; that a county
dominating on all three signals scores exactly 100 and a dominated one exactly 0;
that the composite equals its weighted parts; that the OLS fit recovers a known
line to floating-point precision in both directions; and that stratum aggregates
reconcile county counts and population totals against the rows they came from.

---

## 6. SDLC and CI/CD

`.github/workflows/ci.yml` runs on every push and pull request to `main`, with
in-progress runs cancelled on a new push to the same ref.

**Job 1 — quality gates.** Checkout → `setup-python` with pip caching keyed on
`requirements.txt` → install → `flake8` → `pytest --verbose`. Run as a matrix
across Python 3.11 and 3.12: 3.11 matches the container base image, and 3.12
surfaces deprecations before they become a forced upgrade. `fail-fast` is off so
a failure on one version still reports the other.

**Job 2 — container validation.** Gated behind the quality gates. Builds the
`runtime` stage with Buildx and a GitHub Actions layer cache, then asserts two
properties of the shipped artefact: that `id -u` inside it is not `0`, and that
the suite passes *inside the image* rather than only on the runner. A Dockerfile
that builds is not the same as an image that works.

Local and CI gates are the same commands — a green `make lint test` means the
pipeline agrees.

---

## 7. Design decisions worth noting

- **Ordinal, not categorical, colour.** Endemicity strata are ordered by
  transmission intensity, so they are encoded on a single-hue navy ramp rather
  than a categorical palette. Each stratum keeps its step regardless of which
  strata survive a filter, so narrowing the sidebar never repaints the survivors.
- **No dual-axis charts.** The stratum comparison shows risk and parasitemia as
  paired panels against a shared axis, not as two y-scales on one plot, which
  would let the panel geometry imply a crossover the data does not contain.
- **Population-weighted headline rate.** The national parasitemia KPI is weighted
  by population; an unweighted county mean would let Lamu (161k residents) and
  Nairobi (4.9m) move the national figure by the same amount.
- **`SchemaError` subclasses `ValueError`.** Callers that only care that
  ingestion failed can catch `ValueError`; callers that need to distinguish a
  contract breach from a bad argument can catch the specific type.

---

## 8. Limitations

- **The dataset is a realistic seed extract, not an official release.** It is
  modelled on published KMIS/KHIS patterns and 2019-census-derived population
  projections for demonstration purposes. Do not cite these figures.
- **The index is cohort-relative.** Min-max scaling ranks counties against each
  other, not against an absolute threshold, so scores are not comparable across
  years. A longitudinal deployment would need fixed reference bounds.
- **The actionability term puts a floor under near-zero-transmission counties.**
  A highland county at 0.2% parasitemia still earns most of the 20-point net-gap
  allocation and can outrank a county with ten times its burden. The fixed tier
  cut points contain this — everything under 25 reads *Low* regardless — but the
  raw index is a prioritisation ranking, not a burden measure, at the bottom of
  its range. Conditioning the net-gap term on a minimum transmission threshold
  would remove the artifact at the cost of a discontinuity in the index.
- **The ITN–parasitemia coefficient is ecological and cross-sectional.** It
  reflects programme targeting, not intervention effect, and is labelled as such
  in the dashboard.
- **Weights are expert-assigned, not fitted.** They encode a stated prioritisation
  policy. Changing them changes the ranking; the tests assert structure, so they
  stay valid under a retune.

---

## License

MIT — see [`LICENSE`](LICENSE).
