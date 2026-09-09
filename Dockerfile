# syntax=docker/dockerfile:1
#
# AfyaTrack container image.
#
# Two stages. The builder compiles every dependency to a wheel; the runtime
# installs from that wheel directory with `--no-index`, so no compiler, no VCS
# client, and no pip HTTP cache ever reach the shipped filesystem layer. The
# runtime stage also never runs as root: the application user is created before
# any application file is copied, and every COPY sets its ownership explicitly.

# --------------------------------------------------------------------------
# Stage 1 - build dependency wheels
# --------------------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip wheel --wheel-dir /wheels --requirement requirements.txt

# --------------------------------------------------------------------------
# Stage 2 - runtime
# --------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="AfyaTrack" \
      org.opencontainers.image.description="Subnational malaria surveillance and analytical modeling platform" \
      org.opencontainers.image.source="https://github.com/brianngumbau/afyatrack" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Fixed uid/gid so bind-mounted files keep predictable ownership across hosts.
RUN groupadd --system --gid 1001 afyatrack \
    && useradd --system --uid 1001 --gid afyatrack --no-create-home afyatrack

COPY --from=builder /wheels /wheels
COPY requirements.txt ./
RUN python -m pip install --no-index --find-links=/wheels --requirement requirements.txt \
    && rm -rf /wheels

# Copied as separate layers so an application edit does not invalidate the
# dependency layer above it. The test suite and lint config ship deliberately:
# they let CI run the same gates inside the image that it runs on the host.
COPY --chown=afyatrack:afyatrack setup.cfg ./
COPY --chown=afyatrack:afyatrack data/ ./data/
COPY --chown=afyatrack:afyatrack .streamlit/ ./.streamlit/
COPY --chown=afyatrack:afyatrack src/ ./src/
COPY --chown=afyatrack:afyatrack tests/ ./tests/
COPY --chown=afyatrack:afyatrack app.py ./

USER afyatrack

EXPOSE 8501

# Streamlit's own liveness endpoint; `urllib` avoids installing curl.
HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4)" \
        || exit 1

# CMD rather than ENTRYPOINT so `docker compose run --rm afyatrack pytest -v`
# overrides the command cleanly instead of appending to it.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
