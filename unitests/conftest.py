"""
conftest.py – shared pytest configuration for all unitests.

Must be imported before any matplotlib code. Sets the non-interactive Agg
backend so figures can be built without a display.
"""
import os
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")  # must precede any other matplotlib import

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "GUI"))

# Keep the viewer's "remember last algorithm choice" prefs out of the real
# user config: point them at a fresh (empty) temp file for the whole session.
# Tests that exercise persistence pass their own prefs_path explicitly.
os.environ["RECONSTRUCT3D_PREFS"] = os.path.join(
    tempfile.mkdtemp(prefix="r3d_prefs_"), "algo_params.json")

INPUT_PATH = os.path.join(ROOT, "inputs", "aircraft_input.json")
