"""
Unit and regression test for the BindingSite class in the ratar.encoding module of the ratar package.
"""

from pathlib import Path

import pytest

from ratar.encoding import BindingSite


@pytest.mark.parametrize(
    "mol_file1, mol_file2", [("AAK1_4wsq_altA_chainA.mol2", "AAK1_4wsq_altA_chainB.mol2")]
)
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

    molecule_path1 = Path(__name__).parent / "ratar" / "tests" / "data" / mol_file1
    molecule_path2 = Path(__name__).parent / "ratar" / "tests" / "data" / mol_file2

    bindingsite1 = BindingSite()
    bindingsite2 = BindingSite()
    bindingsite3 = BindingSite()

    bindingsite1.from_file(molecule_path1)
    bindingsite2.from_file(molecule_path1)
    bindingsite3.from_file(molecule_path2)

    assert bindingsite1 == bindingsite2
    assert not (bindingsite1 == bindingsite3)
