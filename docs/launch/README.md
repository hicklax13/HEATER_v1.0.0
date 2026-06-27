# HEATER Public Commercial Launch — operating index

- **Master spec:** [`../superpowers/specs/2026-06-26-heater-public-commercial-launch-program-design.md`](../superpowers/specs/2026-06-26-heater-public-commercial-launch-program-design.md)
- **Evidence registry (source of truth for the score):** [`evidence_registry.yaml`](evidence_registry.yaml)
- **Baseline reports:** [`baseline/`](baseline/)
- **Phase plans:** `../superpowers/plans/2026-06-*-heater-launch-phase*.md`

## How to read the registry

Each requirement row has a `status`: `planned` (not started), `in_progress`,
`passing` (its `verify` command is green), `failing`, `deferred` (intentionally
later, e.g. needs a concept from a later phase), or `waived`. Run
`python -m scripts.launch.evidence_registry --summary` for a status rollup.

## How to refresh the baseline

`python -m scripts.launch.freeze_baseline` writes a timestamped report into `baseline/`.
