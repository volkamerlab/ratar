"""
Unit and regression test for the Shapes class in the ratar.encoding module of the ratar package.
"""

import sys

import pandas as pd

from flatten_dict import flatten
import pytest

from ratar.auxiliary import MoleculeLoader
from ratar.encoding import Shapes


@pytest.mark.parametrize('distances, moment1, moment2, moment3', [
    (
        pd.DataFrame([
            [1.0, 2.0],
            [1.0, 2.0],
            [4.0, 2.0],
            [4.0, 2.0],
            [5.0, 2.0]
        ], columns='dist_c1 dist_c2'.split()),
        pd.Series([3.0, 2.0], index='dist_c1 dist_c2'.split()),
        pd.Series([1.6733, 0.0000], index='dist_c1 dist_c2'.split()),
        pd.Series([-1.0627, 0.0000], index='dist_c1 dist_c2'.split())
    )
])
def test_calc_moments(distances, moment1, moment2, moment3):

    shapes = Shapes()
    moments = shapes._calc_moments(distances)

    print(moments)

    assert all((moments['m1'] - moment1) < 0.0001)
    assert all((moments['m2'] - moment2) < 0.0001)
    assert all((moments['m3'] - moment3) < 0.0001)


@pytest.mark.parametrize('nested_dict, key_order, flat_keys_before, flat_keys_after', [
    (
        {'H': {'O': {'L': {'A': 'halo', 'E': 'helo'}}}},
        [0, 3, 2, 1],
        ['H/O/L/A', 'H/O/L/E'],
        ['H/A/L/O', 'H/E/L/O']
    )
])
def test_reorder_nested_dict_keys(nested_dict, key_order, flat_keys_before, flat_keys_after):

    shapes = Shapes()
    reordered_dict = shapes._reorder_nested_dict_keys(nested_dict, key_order)

    assert sorted(list(flatten(nested_dict, reducer='path').keys())) == flat_keys_before
    assert sorted(list(flatten(reordered_dict, reducer='path').keys())) == flat_keys_after
