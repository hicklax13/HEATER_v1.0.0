"""HEATER mini-PC Yahoo-token relay.

Run every 30 min by Windows Task Scheduler. Refreshes the Yahoo token from this
machine's (residential) IP — which Yahoo accepts, unlike the Railway datacenter IP —
encrypts it, and uploads it to a secret GitHub gist that the Railway server reads.

Env required: HEATER_RELAY_KEY (Fernet key), HEATER_GIST_ID, HEATER_GIST_PAT (gist scope).
Never prints the token or any secret.
"""

import json
import logging
import os
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.token_relay import _fernet, encrypt_token  # noqa: E402
from src.yahoo_api import _AUTH_DIR, _write_token_file, refresh_yahoo_token  # noqa: E402

logger = logging.getLogger("yahoo_token_relay")

TOKEN_FILE = _AUTH_DIR / "yahoo_token.json"
GIST_FILENAME = "heater_yahoo_token.enc"


def main() -> int:
    if not TOKEN_FILE.exists():
        logger.error("No token file at %s — seed it once via the local OAuth flow.", TOKEN_FILE)
        return 1
    try:
        token = json.loads(TOKEN_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.error(
            "Token file at %s is unreadable/corrupt (%s) — reseed via the local OAuth flow.",
            TOKEN_FILE,
            type(exc).__name__,
        )
        return 6

    fresh = refresh_yahoo_token(token)
    if fresh is None:
        logger.error("Refresh failed (see the Yahoo error above).")
        return 2
    if not _write_token_file(fresh):
        logger.warning("Could not refresh local seed token; next run may read a stale refresh_token.")

    f = _fernet()
    if f is None:
        logger.error("HEATER_RELAY_KEY not set / invalid — cannot encrypt.")
        return 3

    gist_id = os.environ.get("HEATER_GIST_ID", "").strip()
    pat = os.environ.get("HEATER_GIST_PAT", "").strip()
    if not (gist_id and pat):
        logger.error("HEATER_GIST_ID / HEATER_GIST_PAT not set — cannot upload.")
        return 4

    ciphertext = encrypt_token(fresh, f)
    try:
        resp = requests.patch(
            f"https://api.github.com/gists/{gist_id}",
            headers={"Authorization": f"token {pat}", "Accept": "application/vnd.github+json"},
            json={"files": {GIST_FILENAME: {"content": ciphertext}}},
            timeout=20,
        )
    except requests.exceptions.RequestException as exc:
        logger.error("Gist upload request failed: %s", type(exc).__name__)
        return 5
    if resp.status_code not in (200, 201):
        logger.error("Gist upload failed: HTTP %s — %s", resp.status_code, (resp.text or "")[:200])
        return 5

    logger.info("Relay OK: token refreshed + uploaded (token_time advanced).")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    raise SystemExit(main())
