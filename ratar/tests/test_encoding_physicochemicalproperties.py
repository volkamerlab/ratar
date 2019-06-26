"""
Unit and regression test for the PhysicoChemicalProperties class in the ratar.encoding module of the ratar package.
"""

import sys

from flatten_dict import flatten
import pytest
from pathlib import Path

from ratar.auxiliary import MoleculeLoader
from ratar.encoding import PhysicoChemicalProperties


@pytest.mark.parametrize('filename, column_names, n_atoms, centroid', [
    (
        'AAK1_4wsq_altA_chainA_reduced.mol2',
        dict(
            zip(
                'ca/z1 ca/z123 pca/z1 pca/z123 pc/z1 pc/z123'.split(),
                [
                    ['z1'],
                    ['z1', 'z2', 'z3'],
                    ['z1'],
                    ['z1', 'z2', 'z3'],
                    ['z1'],
                    ['z1', 'z2', 'z3']]
            )
        ),
        dict(
            zip(
                'ca/z1 ca/z123 pca/z1 pca/z123 pc/z1 pc/z123'.split(),
                [8, 8, 34, 34, 29, 29]
            )
        ),
        dict(
            zip(
                'ca/z1 ca/z123 pca/z1 pca/z123 pc/z1 pc/z123'.split(),
                [
                    [-0.0662],
                    [-0.0663, -1.49, -0.1088],
                    [-0.5888],
                    [-0.5888, -1.4906, -0.3768],
                    [-0.069],
                    [-0.069, -1.431, -0.1814]]
            )
        )
    )
])
def test_get_physicochemicalproperties_from_pmol(filename, column_names, n_atoms, centroid):
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
    molecule_path = Path(sys.path[0]) / 'ratar' / 'tests' / 'data' / filename
    molecule_loader = MoleculeLoader()
    molecule_loader.load_molecule(molecule_path)
    pmol = molecule_loader.get_first_molecule()

    # Set physicochemical properties
    physicochemicalproperties = PhysicoChemicalProperties()
    physicochemicalproperties.get_physicochemicalproperties_from_pmol(pmol)

    pcp_flat = flatten(physicochemicalproperties.data, reducer='path')

    for key, value in pcp_flat.items():
        assert all(value.columns == column_names[key])
        assert value.shape[0] == n_atoms[key]

        for i, column_name in enumerate(value.columns):
            assert abs(value[column_name].mean() - centroid[key][i]) < 0.0001
