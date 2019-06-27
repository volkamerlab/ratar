"""
Unit and regression test for the Points class in the ratar.encoding module of the ratar package.
"""

import sys

from flatten_dict import flatten
from pathlib import Path
import pytest

from ratar.auxiliary import MoleculeLoader
from ratar.encoding import Points


@pytest.mark.parametrize('mol_file1, mol_file2', [
    ('AAK1_4wsq_altA_chainA.mol2', 'AAK1_4wsq_altA_chainB.mol2')
])
def test_points_eq(mol_file1, mol_file2):
    """
    Test __eq__ function for Points class.

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

    obj1 = Points()
    obj2 = Points()
    obj3 = Points()

    obj1.get_points_from_pmol(molecule_loader1.get_first_molecule())
    obj2.get_points_from_pmol(molecule_loader1.get_first_molecule())
    obj3.get_points_from_pmol(molecule_loader2.get_first_molecule())

    assert (obj1 == obj2) is True
    assert (obj1 == obj3) is False


@pytest.mark.parametrize('filename, keys, n_dimensions', [
    (
        'AAK1_4wsq_altA_chainA_reduced.mol2',
        'ca/no ca/z1 ca/z123 pca/no pca/z1 pca/z123 pc/no pc/z1 pc/z123'.split(),
        dict(
            zip(
                'ca/no ca/z1 ca/z123 pca/no pca/z1 pca/z123 pc/no pc/z1 pc/z123'.split(),
                [3, 4, 6, 3, 4, 6, 3, 4, 6]
            )
        )
    )
])
def test_get_points_from_pmol(filename, keys, n_dimensions):
    """
    Test if points are correctly extracted from representatives of a molecule.
    
    Parameters
    ----------
    filename : str
        Name of molecule file.
    keys : list of str
        Flattened keys for different types of representatives and physicochemical properties.
    n_dimensions : list of int
        Number of dimensions for different types of representatives and physicochemical properties.
    """

    # Load molecule
    molecule_path = Path(sys.path[0]) / 'ratar' / 'tests' / 'data' / filename
    molecule_loader = MoleculeLoader()
    molecule_loader.load_molecule(molecule_path)
    pmol = molecule_loader.get_first_molecule()

    # Set points
    points = Points()
    points.get_points_from_pmol(pmol)

    points_flat = flatten(points.data, reducer='path')

    assert sorted(points_flat.keys()) == sorted(keys)

    for key, value in points_flat.items():
        assert value.shape[1] == n_dimensions[key]
