"""
Unit and regression test for the BindingSite class in the ratar.encoding module of the ratar package.
"""

import pytest
from pathlib import Path

from ratar.auxiliary import MoleculeLoader
from ratar.encoding import BindingSite


@pytest.mark.parametrize('mol_file1, mol_file2', [
    ('AAK1_4wsq_altA_chainA.mol2', 'AAK1_4wsq_altA_chainB.mol2')
])
def test_bindingsites_eq(mol_file1, mol_file2):
    """
    Test __eq__ function for BindingSite class.

    Parameters
    ----------
    mol_file1 : str
        Name of file containing the structure for molecule A.
    mol_file2 : str
        Name of file containing the structure for molecule B.

    """

    molecule_path1 = Path(__name__).parent / 'ratar' / 'tests' / 'data' / mol_file1
    molecule_path2 = Path(__name__).parent / 'ratar' / 'tests' / 'data' / mol_file2

    molecule_loader1 = MoleculeLoader(molecule_path1)
    molecule_loader2 = MoleculeLoader(molecule_path2)

    bindingsite1 = BindingSite(molecule_loader1.molecules_list[0])
    bindingsite2 = BindingSite(molecule_loader1.molecules_list[0])
    bindingsite3 = BindingSite(molecule_loader2.molecules_list[0])

    assert (bindingsite1 == bindingsite2)
    assert not (bindingsite1 == bindingsite3)
