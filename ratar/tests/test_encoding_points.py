"""
Unit and regression test for the Points class in the ratar.encoding module of the ratar package.
"""

import sys

from pathlib import Path
import pytest

from ratar.auxiliary import MoleculeLoader
from ratar.encoding import Representatives, Coordinates, PhysicoChemicalProperties, Subsets, Points


@pytest.mark.parametrize('', [
    (

    )
])
def test_get_points():

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
    points = Points
    points.get_points(coordinates, physicochemicalproperties)
    points.get_points_pseudocenter_subsets(coordinates, physicochemicalproperties, subsets)

    assert 0 == 1
