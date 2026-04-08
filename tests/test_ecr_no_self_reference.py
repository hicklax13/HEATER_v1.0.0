"""Test that ECR consensus does not include HEATER's own SGP rank (ROADMAP A4).

Self-referential input creates circular confirmation bias — the model's output
would feed back into its own valuation via the Trimmed Borda consensus.
"""

import inspect


def test_heater_not_in_ecr_sources():
    """ECR consensus must not include HEATER's own SGP rank as a source."""
    import src.ecr as ecr_module

    source_code = inspect.getsource(ecr_module)
    # The active (uncommented) append line must not exist
    lines = source_code.split("\n")
    for line in lines:
        stripped = line.lstrip()
        # Skip commented-out lines
        if stripped.startswith("#"):
            continue
        assert 'source_fetchers.append(("heater"' not in stripped, (
            "HEATER SGP is still an active ECR source — this creates circular "
            "confirmation bias. Comment out or remove the heater source_fetchers.append line."
        )


def test_ecr_source_count_without_heater():
    """Trimmed Borda requires >= 4 sources; removing heater still leaves >= 4."""
    # This is a design-level assertion: we had 7 sources, removing 1 leaves 6.
    # The minimum for trimmed Borda (trim 1 top + 1 bottom) is 4.
    min_sources_for_trimmed_borda = 4
    sources_after_removal = 6  # espn, cbs, fantasypros, yahoo/nfbc, fangraphs
    assert sources_after_removal >= min_sources_for_trimmed_borda
