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

# Shell form so ${PORT:-8501} expands: a bare local `docker run` (no $PORT) still
# binds 8501; on Railway the railway.toml startCommand overrides this CMD and
# uses the injected $PORT. Both are intentional — see the design spec §5.
CMD streamlit run app.py \
    --server.port "${PORT:-8501}" \
    --server.address 0.0.0.0 \
    --server.headless true
