# Security scanning — status & follow-ups (Phase 10 secure SDLC)

**Date:** 2026-06-26 · **Status:** first SAST gate shipped; secret + dependency scanning verified-clean locally, CI gates are follow-ups.

This tracks the Codex "secure SDLC in CI" findings (no SAST / secret / dependency / container scanning, no SBOM).

## Shipped

- **SAST (bandit)** — CI job `security-scan` in `.github/workflows/ci.yml` runs `bandit -r api/ --severity-level high` (pinned `bandit==1.9.4`). The public API surface is **clean at HIGH severity today** (exit 0). The 3 Medium findings are the `f"SELECT {self._COLS} FROM …"` pattern in `api/stores/` — **false positives** (the interpolated part is a hardcoded column constant; the actual values are parameterized with `?`). 11 Low findings are minor. The gate blocks only on genuine HIGH-severity issues for now.

## Verified locally (clean) — CI gates are follow-ups

- **Secret scanning** — ran `detect-secrets` over the tree: **586 potential secrets flagged, ZERO real.** 567 are `web/pnpm-lock.yaml` integrity hashes (not secrets); the other 19 are example templates (`.streamlit/secrets.toml.example`, `alembic.ini`), test fixtures (`tests/test_*token*`, `test_ai_providers*`), and doc examples — all false positives. **Confirms no committed secrets** (consistent with the keys living only in gitignored `.env` / Railway / local config). 
  - *Follow-up:* wire a CI secret-scan gate. Recommended: **gitleaks** GitHub Action (single binary, runs identically in CI; avoids the Windows-dev → Linux-CI path-portability problem that makes a committed `detect-secrets` baseline brittle) with a `.gitleaks.toml` allowlist for the lockfile, `*.example`, and test fixtures. Scan the working tree + PR diff (a full-history scan is a separate, deliberate audit).

- **Dependency CVEs (pip-audit)** — a local run reported "167 vulnerabilities in 46 packages," but that is the **kitchen-sink local env** (Python 3.14 + the full data-science + dev stack), **not** the production `requirements.txt` set on 3.12. The number is misleading and not actionable as-is. (pip-audit also crashed on a Windows cp1252 encoding quirk printing a CVE description — use `--format json` there.)
  - *Follow-up:* a real **dependency triage** — `pip-audit -r requirements.txt` on Python 3.12, then fix the fixable (bump pins) and `--ignore-vuln` the unfixable transitive CVEs with documented justification, then add a CI job (report-only first, blocking after triage).

## Remaining secure-SDLC follow-ups (Phase 10)
- Tighten bandit to `--severity-level medium` after adding `# nosec B608` (with justification) to the column-list f-string sites in `api/stores/`.
- Expand SAST scope from `api/` to `src/` (larger, legacy; triage separately).
- SBOM generation (e.g., `cyclonedx-bom`) as a release artifact.
- Container image scanning (once the API Dockerfile is the deploy artifact).
- Move secrets to a managed secret store + rotation (Phase 10 credential work).
