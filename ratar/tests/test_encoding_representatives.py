"""
Unit and regression test for the Representatives class in the ratar.encoding module of the ratar package.
"""

import sys

import pytest
from pathlib import Path

from ratar.auxiliary import MoleculeLoader
from ratar.encoding import Representatives


@pytest.mark.parametrize('filename, column_names, n_atoms, centroid', [
    (
        'AAK1_4wsq_altA_chainA_reduced.mol2',
        {
            'ca': 'atom_id atom_name res_id res_name subst_name x y z charge'.split(),
            'pca': 'atom_id atom_name res_id res_name subst_name x y z charge pc_type pc_id pc_atom_id'.split(),
            'pc': 'atom_id atom_name res_id res_name subst_name x y z charge pc_type pc_id pc_atom_id'.split()
        },
        {
            'ca': 8,
            'pca': 34,
            'pc': 29
        },
        {
            'ca': [6.2681, 11.9717, 42.4514],
            'pca': [5.6836, 12.9039, 43.9326],
            'pc': [5.8840, 12.5871, 43.3804]
        }
    )
])
def test_get_representatives(filename, column_names, n_atoms, centroid):
    """
    Test if representatives are correctly extracted from representatives of a molecule.

    Parameters
    ----------
    filename : str
        Name of molecule file.
    column_names : dict list of str
        List of molecule DataFrame column names for each representatives type.
    n_atoms : dict of int
        Number of atoms for each representatives type.
    centroid : dict of list of float
        3D coordinates of molecule centroid for each representatives type.
    """

    # Load molecule
    molecule_path = Path(sys.path[0]) / 'ratar' / 'tests' / 'data' / filename
    molecule_loader = MoleculeLoader()
    molecule_loader.load_molecule(molecule_path)
    molecule = molecule_loader.pmols[0].df

    # Set representatives
    representatives = Representatives()
    representatives.get_representatives(molecule)

    for key, value in representatives.data.items():
        assert all(value.columns == column_names[key])
        assert value.shape[0] == n_atoms[key]
        assert abs(value['x'].mean() - centroid[key][0]) < 0.0001
        assert abs(value['y'].mean() - centroid[key][1]) < 0.0001
        assert abs(value['z'].mean() - centroid[key][2]) < 0.0001
