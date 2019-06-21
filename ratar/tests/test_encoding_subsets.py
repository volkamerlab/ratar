"""
Unit and regression test for the Subsets class in the ratar.encoding module of the ratar package.
"""

import sys

from pathlib import Path
import pytest

from ratar.auxiliary import MoleculeLoader
from ratar.encoding import Representatives, Subsets


@pytest.mark.parametrize('filename, subsets_names, example_indices', [
    (
        'AAK1_4wsq_altA_chainA_reduced.mol2',
        'H HBD AR HBA'.split(),
        ('pc', 'HBA', [5, 13, 14, 20, 36, 55, 65, 76, 83, 89])
    )
])
def test_get_pseudocenter_subsets_indices(filename, subsets_names, example_indices):
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

    # Set representatives
    representatives = Representatives()
    representatives.get_representatives(molecule)

    # Set subsets
    subsets = Subsets()
    subsets.get_pseudocenter_subsets_indices(representatives)

    assert list(subsets.pseudocenters.keys()).sort() == subsets_names.sort()
    assert list(subsets.pseudocenter_atoms.keys()).sort() == subsets_names.sort()
    assert subsets.data_pseudocenter_subsets[example_indices[0]][example_indices[1]] == example_indices[2]
