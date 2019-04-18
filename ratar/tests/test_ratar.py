"""
Unit and regression test for the ratar package.
"""

# Import package, test suite, and other packages as needed
import ratar
import pytest
import sys

def test_ratar_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "ratar" in sys.modules
