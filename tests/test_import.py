"""
Smoke test for the freefield package.

This test checks whether the package can be found and imported in a clean
Python environment. It should not require lab hardware or running experiments.
"""

import importlib
import importlib.util


def test_freefield_package_can_be_found():
    """Check that Python can locate the freefield package."""
    spec = importlib.util.find_spec("freefield")

    assert spec is not None, "Python could not find the freefield package."


def test_freefield_package_can_be_imported():
    """Check that the top-level freefield package can be imported."""
    freefield = importlib.import_module("freefield")

    assert freefield is not None
    assert freefield.__name__ == "freefield"