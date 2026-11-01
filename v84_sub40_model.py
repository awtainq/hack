"""Reproducible entry point for the original V84 forecast.

Run: .venv/bin/python v84_sub40_model.py

Implementation: v84_refactored.py
Frozen parameters and input provenance: output/v84_audit/snapshot/manifest.json
Original source: output/v84_audit/original_v84_sub40_model.py

V84 calibrates previous forecasts against leaderboard feedback. Its reported
Public RMSLE is 0.39140; this is not an independently trained historical model.
For historical-only training, use v85_historical_model.py and evaluate_v85.py.
"""
from v84_refactored import main


if __name__ == "__main__":
    main()
