"""
Unit and regression test for the ratar.cli module of the ratar package.
"""

import sys

import pytest
from pathlib import Path
import pickle

from ratar.auxiliary import MoleculeLoader
from ratar.encoding import BindingSite


def test_ratar_imported():
    """
    Sample test, will always pass so long as import statement worked
    """
    assert 'ratar' in sys.modules

