"""
Unit and regression test for the Subsets class in the ratar.encoding module of the ratar package.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ratar.auxiliary import MoleculeLoader
from ratar.similarity import calculate_similarity, get_similarity_all_against_all, get_similarity_pairs


@pytest.mark.parametrize('fingerprint1, fingerprint2, measure, similarity', [
    (
        pd.DataFrame([[1, 2], [4, 1]]),
        pd.DataFrame([[0, 0], [0, 0]]),
        'modified_manhattan',
        0.3333
    ),
    (
        pd.DataFrame([[0, 0], [0, 0]]),
        pd.DataFrame([[0, 0], [0, 0]]),
        'modified_manhattan',
        1
    ),
    (
        [1, 2, 4, 1],
        [0, 0, 0, 0],
        'modified_manhattan',
        0.3333
    )
])
def test_calculate_similarity(fingerprint1, fingerprint2, similarity, measure):
    """
    Test if similarity is correctly calculated for two arrays or pandas.DataFrames.

    Parameters
    ----------
    fingerprint1 : 1D array-like or pandas.DataFrame
        Fingerprint for molecule.
    fingerprint2 : 1D array-like or pandas.DataFrame
        Fingerprint for molecule.
    measure : str
        Similarity measurement method.
    similarity : float
        Similarity value.
    """

    assert np.isclose(calculate_similarity(fingerprint1, fingerprint2, measure), similarity, rtol=1e-04)


@pytest.mark.parametrize('fingerprint1, fingerprint2, measure', [
    (
        [1, 1, 1],
        [0, 0, 0, 0],
        'modified_manhattan'
    ),
    (
        [1, 1, 1, 1],
        [0, 0, 0, 0],
        'unknown'
    )
])
def test_calculate_similarity_exception(fingerprint1, fingerprint2, measure):
    """
    Test if function exceptions are raised correctly.

    Parameters
    ----------
    fingerprint1 : 1D array-like or pandas.DataFrame
        Fingerprint for molecule.
    fingerprint2 : 1D array-like or pandas.DataFrame
        Fingerprint for molecule.
    measure : str
        Similarity measurement method.
    """
    with pytest.raises(ValueError):
        assert calculate_similarity(fingerprint1, fingerprint2, measure)


@pytest.mark.parametrize('', [
    ()
])
def ttest_get_similarity_all_against_all():
    """

    """
    get_similarity_all_against_all()


@pytest.mark.parametrize('pairs, encoded_molecules_path', [
    (
        pd.DataFrame({'molecule1': ['AAK1_4wsq_altA_chainA'],
                      'molecule2': ['AAK1_4wsq_altA_chainB']}),
        '%.mol2'
    )
])
def ttest_get_similarity_pairs(pairs, encoded_molecules_path):
    """

    """

    assert get_similarity_pairs(pairs, encoded_molecules_path)
