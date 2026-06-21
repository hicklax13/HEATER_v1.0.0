# HEATER — Railway deploy image for the FastAPI API (the React product's backend).
# Mirrors the root Dockerfile (python:3.12-slim = CI parity, identical dependency
# install) but boots the API via start-api.sh (dedicated scheduler -> uvicorn)
# instead of Streamlit. Point the Railway API service's Dockerfile Path at this file
# (the root Dockerfile stays the Streamlit image).
FROM python:3.12-slim

WORKDIR /app

# System deps: gcc/g++ for any source-built wheels (same as the root image).
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer cache). IDENTICAL to the root Dockerfile, incl.
# the --no-deps yfpy/streamlit-oauth step (python-dotenv pin conflict).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --no-deps "yfpy>=17.0" "streamlit-oauth>=0.1.14"

# App code.
COPY . .

EXPOSE 8000

# start-api.sh reads $PORT, seeds the Yahoo token (first boot), launches the
# scheduler, then execs uvicorn (api.main is a FACTORY: --factory).
CMD ["sh", "start-api.sh"]
