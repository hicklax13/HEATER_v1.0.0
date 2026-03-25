"""
HEATER Validation Harness

Infrastructure to answer the question every analytical module currently dodges:
"Does this recommendation actually produce better fantasy baseball outcomes?"

Architecture:
    calibration_data.py   — Fetches historical drafts, trades, matchups from Yahoo
    survival_calibrator.py — Predicted vs actual pick availability
    trade_calibrator.py    — Trade grades vs actual post-trade performance
    lineup_calibrator.py   — Optimized lineups vs actual weekly H2H results
    constant_optimizer.py  — Data-driven replacement for every magic number
    report.py              — Human-readable validation reports

Usage:
    from src.validation import run_full_validation
    report = run_full_validation(yahoo_client, seasons=[2025])
    report.print_summary()
    report.save("docs/validation/2025-report.md")
"""
