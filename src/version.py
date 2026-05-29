"""Running app version, stamped onto feedback so the admin can reproduce context.

Defaults to the current local/CLAUDE.md folder version ("1.0.1"). Override at
deploy time with the HEATER_APP_VERSION env var (e.g. a Railway build tag) so
feedback rows record exactly which build the user was on.
"""

import os

APP_VERSION = os.environ.get("HEATER_APP_VERSION") or "1.0.1"
