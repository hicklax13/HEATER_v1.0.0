#!/usr/bin/env sh
# HEATER container entrypoint (#9, 2026-06-07).
#
# Goal: data refreshes right after a deploy WITHOUT waiting for the first browser.
# Streamlit only runs app.py's main() — which starts the in-process scheduler — on
# a real session/websocket connect, so a freshly deployed container would otherwise
# serve stale data until someone opened it. Here we launch a DEDICATED scheduler
# process at boot (the sole SQLite writer) and then exec Streamlit.
#
# Single-writer invariant (single replica): HEATER_SCHEDULER_BOOT tells every
# Streamlit session process to stay read-only (see src/scheduler.py
# _is_boot_managed_reader); only the dedicated process below marks itself the owner
# (HEATER_SCHEDULER_IS_OWNER). The dedicated writer is gated on MULTI_USER so a v1
# / local `docker run` (flag unset) is byte-for-byte unchanged — no thread, the app
# bootstraps per-session as before.
set -eu

PORT="${PORT:-8501}"
export HEATER_SCHEDULER_BOOT=1

if [ "${MULTI_USER:-}" = "1" ]; then
    # Supervise the dedicated writer: if it ever exits, restart it (best-effort)
    # so a crash never leaves the app with no refresher. `|| true` + sleep avoid a
    # tight crash-loop. Normally `python -m src.scheduler` blocks forever.
    (
        while true; do
            HEATER_SCHEDULER_IS_OWNER=1 python -m src.scheduler || true
            sleep 10
        done
    ) &
fi

# exec so Streamlit becomes the process Railway's healthcheck + restart policy
# watch; the backgrounded scheduler rides alongside in the same replica.
exec streamlit run app.py \
    --server.port "$PORT" \
    --server.address 0.0.0.0 \
    --server.headless true
