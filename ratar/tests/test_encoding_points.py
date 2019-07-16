"""
Unit and regression test for the Points class in the ratar.encoding module of the ratar package.
"""

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

    molecule_path1 = Path(__name__).parent / 'ratar' / 'tests' / 'data' / mol_file1
    molecule_path2 = Path(__name__).parent / 'ratar' / 'tests' / 'data' / mol_file2

    molecule_loader1 = MoleculeLoader(molecule_path1)
    molecule_loader2 = MoleculeLoader(molecule_path2)

    points1 = Points()
    points2 = Points()
    points3 = Points()

    points1.from_molecule(molecule_loader1.molecules_list[0])
    points2.from_molecule(molecule_loader1.molecules_list[0])
    points3.from_molecule(molecule_loader2.molecules_list[0])

    assert (points1 == points2)
    assert not (points1 == points3)


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
def test_from_molecule(filename, keys, n_dimensions):
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
    molecule_path = Path(__name__).parent / 'ratar' / 'tests' / 'data' / filename
    molecule_loader = MoleculeLoader(molecule_path)
    molecule = molecule_loader.molecules_list[0]

    # Set points
    points = Points()
    points.from_molecule(molecule)

    points_flat = flatten(points.data, reducer='path')

    assert sorted(points_flat.keys()) == sorted(keys)

    for key, value in points_flat.items():
        assert value.shape[1] == n_dimensions[key]
