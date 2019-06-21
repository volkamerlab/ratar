"""
Unit and regression test for the Points class in the ratar.encoding module of the ratar package.
"""

import sys

from flatten_dict import flatten
from pathlib import Path
import pytest

from ratar.auxiliary import MoleculeLoader
from ratar.encoding import Representatives, Coordinates, PhysicoChemicalProperties, Subsets, Points


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
def test_get_points(filename, keys, n_dimensions):
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
    molecule = molecule_loader.get_first_molecule()

    # Set representatives
    representatives = Representatives()
    representatives.get_representatives(molecule)

    # Set coordinates
    coordinates = Coordinates()
    coordinates.get_coordinates(representatives)

    # Set physicochemical properties
    physicochemicalproperties = PhysicoChemicalProperties()
    physicochemicalproperties.get_physicochemicalproperties(representatives)

    # Set subsets
    subsets = Subsets()
    subsets.get_pseudocenter_subsets_indices(representatives)

    # Set points
    points = Points()
    points.get_points(coordinates, physicochemicalproperties)
    points.get_points_pseudocenter_subsets(subsets)

    points_flat = flatten(points.data, reducer='path')

    assert sorted(points_flat.keys()) == sorted(keys)

    for key, value in points_flat.items():
        assert value.shape[1] == n_dimensions[key]
