"""
Unit and regression test for the Representatives class in the ratar.encoding module of the ratar package.
"""

import sys

import pytest
from pathlib import Path

from ratar.auxiliary import MoleculeLoader
from ratar.encoding import Representatives


@pytest.mark.parametrize('mol_file1, mol_file2', [
    ('AAK1_4wsq_altA_chainA.mol2', 'AAK1_4wsq_altA_chainB.mol2')
])
def test_representatives_eq(mol_file1, mol_file2):
    """
    Test __eq__ function for Representatives class.

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

    obj1 = Representatives()
    obj2 = Representatives()
    obj3 = Representatives()

    obj1.get_representatives_from_molecule(molecule_loader1.get_first_molecule())
    obj2.get_representatives_from_molecule(molecule_loader1.get_first_molecule())
    obj3.get_representatives_from_molecule(molecule_loader2.get_first_molecule())

    assert (obj1 == obj2) is True
    assert (obj1 == obj3) is False


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
def test_get_representatives_from_molecule(filename, column_names, n_atoms, centroid):
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
    molecule = molecule_loader.get_first_molecule()

    # Set representatives
    representatives = Representatives()
    representatives.get_representatives_from_molecule(molecule)

    for key, value in representatives.data.items():
        assert all(value.columns == column_names[key])
        assert value.shape[0] == n_atoms[key]
        assert abs(value['x'].mean() - centroid[key][0]) < 0.0001
        assert abs(value['y'].mean() - centroid[key][1]) < 0.0001
        assert abs(value['z'].mean() - centroid[key][2]) < 0.0001


@pytest.mark.parametrize('filename', [
    (
        'AAK1_4wsq_altA_chainA_reduced.mol2'
    )
])
def test_get_ca_datatypes(filename):

    # Load molecule
    molecule_path = Path(sys.path[0]) / 'ratar' / 'tests' / 'data' / filename
    molecule_loader = MoleculeLoader()
    molecule_loader.load_molecule(molecule_path)
    molecule = molecule_loader.get_first_molecule()

    repres = Representatives()
    ca = repres._get_ca(molecule.df)

    datatypes = {
        'atom_id': int,
        'atom_name': object,
        'res_id': object,
        'res_name': object,
        'subst_name': object,
        'x': float,
        'y': float,
        'z': float,
        'charge': float
    }

    for index, datatype in ca.dtypes.items():
        assert datatype == datatypes[index]


@pytest.mark.parametrize('filename', [
    (
        'AAK1_4wsq_altA_chainA_reduced.mol2'
    )
])
def test_get_pca_datatypes(filename):

    # Load molecule
    molecule_path = Path(sys.path[0]) / 'ratar' / 'tests' / 'data' / filename
    molecule_loader = MoleculeLoader()
    molecule_loader.load_molecule(molecule_path)
    molecule = molecule_loader.get_first_molecule()

    repres = Representatives()
    pca = repres._get_pca(molecule.df)

    datatypes = {
        'atom_id': int,
        'atom_name': object,
        'res_id': object,
        'res_name': object,
        'subst_name': object,
        'x': float,
        'y': float,
        'z': float,
        'charge': float,
        'pc_type': object,
        'pc_id': object,
        'pc_atom_id': object
    }

    for index, datatype in pca.dtypes.items():
        assert datatype == datatypes[index]


@pytest.mark.parametrize('filename', [
    (
        'AAK1_4wsq_altA_chainA_reduced.mol2'
    )
])
def test_get_pca_pc_datatypes(filename):

    # Load molecule
    molecule_path = Path(sys.path[0]) / 'ratar' / 'tests' / 'data' / filename
    molecule_loader = MoleculeLoader()
    molecule_loader.load_molecule(molecule_path)
    molecule = molecule_loader.get_first_molecule()

    repres = Representatives()
    pc = repres._get_pc(molecule.df)

    datatypes = {
        'atom_id': object,
        'atom_name': object,
        'res_id': object,
        'res_name': object,
        'subst_name': object,
        'x': float,
        'y': float,
        'z': float,
        'charge': float,
        'pc_type': object,
        'pc_id': object,
        'pc_atom_id': object
    }

    for index, datatype in pc.dtypes.items():
        assert datatype == datatypes[index]
