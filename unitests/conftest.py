"""
conftest.py – shared pytest configuration for all unitests.

Must be imported before any matplotlib code. Sets the non-interactive Agg
backend so figures can be built without a display.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")  # must precede any other matplotlib import

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "GUI"))

INPUT_PATH = os.path.join(ROOT, "inputs", "aircraft_input.json")
