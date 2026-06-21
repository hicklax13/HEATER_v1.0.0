#!/usr/bin/env sh
# HEATER API container entrypoint (the React product's backend on Railway).
#
# Mirrors start.sh, but for the FastAPI API instead of Streamlit. It is a
# SELF-CONTAINED service: separate Railway services cannot share a filesystem, so
# this container runs its OWN dedicated refresh scheduler (the sole SQLite writer)
# to warm + refresh data on its OWN volume (data/ holding draft_tool.db +
# api_state.db + yahoo_token.json), then execs uvicorn. The existing Streamlit
# service keeps running separately as the fallback. (At M4 both move to a shared
# Postgres and this per-service duplication goes away.)
set -eu

PORT="${PORT:-8000}"
# The dedicated scheduler only warms data under MULTI_USER (see src/scheduler.py
# __main__). The API IS the multi-user backend, so default it on.
export MULTI_USER="${MULTI_USER:-1}"
export HEATER_SCHEDULER_BOOT=1

# Seed the Yahoo token on FIRST boot only (empty volume). The app refreshes the
# token and writes it back to the volume thereafter, so re-writing the (now stale)
# env value on every boot would clobber the fresh token — hence the -f guard.
if [ ! -f data/yahoo_token.json ] && [ -n "${YAHOO_TOKEN_JSON:-}" ]; then
    mkdir -p data
    printf '%s' "$YAHOO_TOKEN_JSON" > data/yahoo_token.json
fi

# Dedicated writer, supervised (restart if it ever exits) — same shape as start.sh.
if [ "${MULTI_USER:-}" = "1" ]; then
    (
        while true; do
            HEATER_SCHEDULER_IS_OWNER=1 python -m src.scheduler || true
            sleep 10
        done
    ) &
fi

# exec uvicorn so it is the process Railway's healthcheck (/healthz) + restart
# policy watch; the backgrounded scheduler rides alongside in the same replica.
exec python -m uvicorn api.main:create_app --factory --host 0.0.0.0 --port "$PORT"
