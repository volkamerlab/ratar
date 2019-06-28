"""
Unit and regression test for the Subsets class in the ratar.encoding module of the ratar package.
"""

import sys

from pathlib import Path
import pytest

from ratar.auxiliary import MoleculeLoader
from ratar.encoding import Subsets


@pytest.mark.parametrize('mol_file1, mol_file2', [
    ('AAK1_4wsq_altA_chainA.mol2', 'AAK1_4wsq_altA_chainB.mol2')
])
def test_subsets_eq(mol_file1, mol_file2):
    """
    Test __eq__ function for Subsets class.

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

    obj1 = Subsets()
    obj2 = Subsets()
    obj3 = Subsets()

    obj1.get_pseudocenter_subsets_indices_from_molecule(molecule_loader1.get_first_molecule())
    obj2.get_pseudocenter_subsets_indices_from_molecule(molecule_loader1.get_first_molecule())
    obj3.get_pseudocenter_subsets_indices_from_molecule(molecule_loader2.get_first_molecule())

    assert (obj1 == obj2) is True
    assert (obj1 == obj3) is False


@pytest.mark.parametrize('filename, subsets_names, example_indices', [
    (
        'AAK1_4wsq_altA_chainA_reduced.mol2',
        'H HBD AR HBA'.split(),
        ('pc', 'HBA', [5, 13, 14, 20, 36, 55, 65, 76, 83, 89])
    )
])
def test_get_pseudocenter_subsets_indices_from_molecule(filename, subsets_names, example_indices):
    """
    Test if pseudocenter subset indices are extracted correctly.

    Parameters
    ----------
    filename : str
        Name of molecule file.
    subsets_names : list of str
        Names of subsets.
    example_indices : tuple
        List of atom indices for example representatives and subset type.
    """

    # Load molecule
    molecule_path = Path(sys.path[0]) / 'ratar' / 'tests' / 'data' / filename
    molecule_loader = MoleculeLoader()
    molecule_loader.load_molecule(molecule_path)
    molecule = molecule_loader.get_first_molecule()

    # Set subsets
    subsets = Subsets()
    subsets.get_pseudocenter_subsets_indices_from_molecule(molecule)

    assert list(subsets.pseudocenters.keys()).sort() == subsets_names.sort()
    assert list(subsets.pseudocenter_atoms.keys()).sort() == subsets_names.sort()
    assert subsets.data_pseudocenter_subsets[example_indices[0]][example_indices[1]] == example_indices[2]
