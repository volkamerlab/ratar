"""
Unit and regression test for the BindingSite class in the ratar.encoding module of the ratar package.
"""

import sys

import pytest
from pathlib import Path

from ratar.auxiliary import MoleculeLoader
from ratar.encoding import BindingSite


@pytest.mark.parametrize('mol_file1, mol_file2', [
    ('AAK1_4wsq_altA_chainA.mol2', 'AAK1_4wsq_altA_chainB.mol2')
])
def ttest_bindingsites_eq(mol_file1, mol_file2):
    """
    Test __eq__ functions for encoding classes.

    Parameters
    ----------
    mol_file1 : str
        Name of file containing the structure for molecule A.
    mol_file2 : str
        Name of file containing the structure for molecule B.

    """

    molecule_path1 = Path(sys.path[0]) / 'ratar' / 'tests' / 'data' / mol_file1
    molecule_path2 = Path(sys.path[0]) / 'ratar' / 'tests' / 'data' / mol_file2

    molecule_loader1 = MoleculeLoader()
    molecule_loader2 = MoleculeLoader()

    molecule_loader1.load_molecule(molecule_path1)
    molecule_loader2.load_molecule(molecule_path2)

    bs1 = BindingSite(molecule_loader1.pmols[0])
    bs2 = BindingSite(molecule_loader1.pmols[0])
    bs3 = BindingSite(molecule_loader2.pmols[0])

    assert (bs1 == bs2) is True
    assert (bs1 == bs3) is False
