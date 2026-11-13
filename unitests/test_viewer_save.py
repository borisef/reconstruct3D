"""Tests for Save JSON functionality in AircraftViewer."""
import json
import os

import pytest
import matplotlib.pyplot as plt

import aircraft_io as aio
from viewer import AircraftViewer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT = os.path.join(ROOT, "inputs", "aircraft_input.json")


@pytest.fixture
def viewer():
    v = AircraftViewer(INPUT)
    v.build_figure()
    yield v
    plt.close(v.fig)


def test_save_writes_exact_doc(viewer, tmp_path):
    out = str(tmp_path / "out.json")
    viewer._ask_save = lambda: out
    viewer._on_save(None)
    saved = aio.load_json(out)
    assert saved == viewer.doc

def test_save_updates_path(viewer, tmp_path):
    out = str(tmp_path / "new.json")
    viewer._ask_save = lambda: out
    viewer._on_save(None)
    assert viewer.path == out

def test_save_updates_status_text(viewer, tmp_path):
    out = str(tmp_path / "s.json")
    viewer._ask_save = lambda: out
    viewer._on_save(None)
    assert "[saved]" in viewer._status.get_text()

def test_save_cancel_does_not_change_path(viewer):
    old_path = viewer.path
    viewer._ask_save = lambda: ""   # simulate cancel
    viewer._on_save(None)
    assert viewer.path == old_path

def test_save_cancel_does_not_set_saved_status(viewer):
    viewer._ask_save = lambda: ""
    viewer._on_save(None)
    assert "[saved]" not in viewer._status.get_text()

def test_save_cancel_none_does_not_change_path(viewer):
    old_path = viewer.path
    viewer._ask_save = lambda: None   # another cancel form
    viewer._on_save(None)
    assert viewer.path == old_path

def test_save_produces_valid_json(viewer, tmp_path):
    out = str(tmp_path / "valid.json")
    viewer._ask_save = lambda: out
    viewer._on_save(None)
    with open(out) as f:
        json.load(f)   # must not raise

def test_save_after_solve_contains_computed(viewer, tmp_path):
    from main import solve_document
    solve_document(viewer.doc, source="GT")
    out = str(tmp_path / "solved.json")
    viewer._ask_save = lambda: out
    viewer._on_save(None)
    saved = aio.load_json(out)
    assert saved["Coordinates_3D"]["P1_tail"]["Computed1"] is not None
