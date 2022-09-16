"""
Unit and regression test for the som_gnn package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import som_gnn


def test_som_gnn_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "som_gnn" in sys.modules
