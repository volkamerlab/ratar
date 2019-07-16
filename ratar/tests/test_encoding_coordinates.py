"""
Unit and regression test for the Coordinates class in the ratar.encoding module of the ratar package.
"""

import numpy as np
import pytest
from pathlib import Path

from ratar.auxiliary import MoleculeLoader
from ratar.encoding import Coordinates


@pytest.mark.parametrize('mol_file1, mol_file2', [
    ('AAK1_4wsq_altA_chainA.mol2', 'AAK1_4wsq_altA_chainB.mol2')
])
def test_coordinates_eq(mol_file1, mol_file2):
    """
    Test __eq__ function for Coordinates class.

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

    coordinates1 = Coordinates()
    coordinates2 = Coordinates()
    coordinates3 = Coordinates()

    coordinates1.from_molecule(molecule_loader1.get_first_molecule())
    coordinates2.from_molecule(molecule_loader1.get_first_molecule())
    coordinates3.from_molecule(molecule_loader2.get_first_molecule())

    assert (coordinates1 == coordinates2)
    assert not (coordinates1 == coordinates3)


@pytest.mark.parametrize('filename, column_names, n_atoms, centroid', [
    (
        'AAK1_4wsq_altA_chainA_reduced.mol2',
        'x y z'.split(),
        dict(
            zip(
                'ca pca pc'.split(),
                [8, 34, 29]
            )
        ),
        dict(
            zip(
                'ca pca pc'.split(),
                [
                    [6.2681, 11.9717, 42.4514],
                    [5.6836, 12.9039, 43.9326],
                    [5.8840, 12.5871, 43.3804]
                ]
            )
        )

    )
])
def test_from_molecule(filename, column_names, n_atoms, centroid):
    """
    Test if coordinates are correctly extracted from representatives of a molecule.

    Parameters
    ----------
    filename : str
        Name of molecule file.
    column_names : list of str
        List of molecule DataFrame columns.
    n_atoms : dict of int
        Number of atoms for each representatives type.
    centroid : dict of list of float
        3D coordinates of molecule centroid for each representatives type.
    """

    # Load molecule
    molecule_path = Path(__name__).parent / 'ratar' / 'tests' / 'data' / filename
    molecule_loader = MoleculeLoader(molecule_path)
    molecule = molecule_loader.get_first_molecule()

    # Set coordinates
    coordinates = Coordinates()
    coordinates.from_molecule(molecule)

    for key, value in coordinates.data.items():
        assert all(value.columns == column_names)
        assert value.shape[0] == n_atoms[key]
        assert np.isclose(value['x'].mean(), centroid[key][0], rtol=1e-04)
        assert np.isclose(value['y'].mean(), centroid[key][1], rtol=1e-04)
        assert np.isclose(value['z'].mean(), centroid[key][2], rtol=1e-04)
