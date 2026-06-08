# HEATER — Railway deploy image (v2 Plan 4).
# Python 3.12-slim matches CI (the app is tested on 3.12 sharded).
FROM python:3.12-slim

WORKDIR /app

# System deps: gcc/g++ for any source-built wheels (scipy/numpy ship wheels, but
# pulp/arviz transitive builds occasionally need a compiler on slim).
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer cache: deps change less often than app code).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --no-deps "yfpy>=17.0" "streamlit-oauth>=0.1.14"

# App code.
COPY . .

EXPOSE 8501

# Boot via start.sh (#9): it launches the dedicated refresh scheduler — so data
# warms at container boot without waiting for a browser session — then execs
# Streamlit (binding the injected $PORT, headless, all interfaces). railway.toml's
# startCommand points at the same script, so local `docker run` and Railway behave
# identically. start.sh handles ${PORT:-8501} and the MULTI_USER gate internally.
CMD ["sh", "start.sh"]
