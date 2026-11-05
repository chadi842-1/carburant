FROM python:3.12-slim

# libgomp for scikit-learn; git is needed by dbt deps resolution in some setups
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY dbt ./dbt
COPY eval ./eval
RUN pip install --no-cache-dir -e .

# Build the warehouse at image build time on the synthetic feed (about a minute).
# For real data, mount a volume at /data and run `carburant ingest --url ...` instead.
ENV CARBURANT_HOME=/data
RUN carburant run --stations 1000 --days 120

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
    CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "carburant.api:app", "--host", "0.0.0.0", "--port", "8000"]
