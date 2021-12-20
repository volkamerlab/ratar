"""
Unit and regression test for the Representatives class in the ratar.encoding module of the ratar package.
"""

from pathlib import Path

import numpy as np
import pytest

from ratar.auxiliary import MoleculeLoader
from ratar.encoding import Representatives


@pytest.mark.parametrize(
    "mol_file1, mol_file2", [("AAK1_4wsq_altA_chainA.mol2", "AAK1_4wsq_altA_chainB.mol2")]
)
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

    molecule_path1 = Path(__name__).parent / "ratar" / "tests" / "data" / mol_file1
    molecule_path2 = Path(__name__).parent / "ratar" / "tests" / "data" / mol_file2

    molecule_loader1 = MoleculeLoader(molecule_path1)
    molecule_loader2 = MoleculeLoader(molecule_path2)

    representatives1 = Representatives()
    representatives2 = Representatives()
    representatives3 = Representatives()

    representatives1.from_molecule(molecule_loader1.molecules[0])
    representatives2.from_molecule(molecule_loader1.molecules[0])
    representatives3.from_molecule(molecule_loader2.molecules[0])

    assert representatives1 == representatives2
    assert not (representatives1 == representatives3)


@pytest.mark.parametrize(
    "filename, column_names, n_atoms, centroid",
    [
        (
            "AAK1_4wsq_altA_chainA_reduced.mol2",
            {
                "ca": "atom_id atom_name res_id res_name subst_name x y z charge".split(),
                "pca": "atom_id atom_name res_id res_name subst_name x y z charge pc_type pc_id pc_atom_id".split(),
                "pc": "atom_id atom_name res_id res_name subst_name x y z charge pc_type pc_id pc_atom_id".split(),
            },
            {"ca": 8, "pca": 34, "pc": 29},
            {
                "ca": [6.2681, 11.9717, 42.4514],
                "pca": [5.6836, 12.9039, 43.9326],
                "pc": [5.8840, 12.5871, 43.3804],
            },
        )
    ],
)
def test_from_molecule(filename, column_names, n_atoms, centroid):
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
    molecule_path = Path(__name__).parent / "ratar" / "tests" / "data" / filename
    molecule_loader = MoleculeLoader(molecule_path)
    molecule = molecule_loader.molecules[0]

    # Set representatives
    representatives = Representatives()
    representatives.from_molecule(molecule)

    for key, value in representatives.data.items():
        assert all(value.columns == column_names[key])
        assert value.shape[0] == n_atoms[key]
        assert np.isclose(value["x"].mean(), centroid[key][0], rtol=1e-04)
        assert np.isclose(value["y"].mean(), centroid[key][1], rtol=1e-04)
        assert np.isclose(value["z"].mean(), centroid[key][2], rtol=1e-04)


@pytest.mark.parametrize("filename", [("AAK1_4wsq_altA_chainA_reduced.mol2")])
def test_get_ca_datatypes(filename):

    # Load molecule
    molecule_path = Path(__name__).parent / "ratar" / "tests" / "data" / filename
    molecule_loader = MoleculeLoader(molecule_path)
    molecule = molecule_loader.molecules[0]

    repres = Representatives()
    ca = repres._get_ca(molecule.df)

    datatypes = {
        "atom_id": int,
        "atom_name": object,
        "res_id": object,
        "res_name": object,
        "subst_name": object,
        "x": float,
        "y": float,
        "z": float,
        "charge": float,
    }

    for index, datatype in ca.dtypes.items():
        assert datatype == datatypes[index]


@pytest.mark.parametrize("filename", [("AAK1_4wsq_altA_chainA_reduced.mol2")])
def test_get_pca_datatypes(filename):

    # Load molecule
    molecule_path = Path(__name__).parent / "ratar" / "tests" / "data" / filename
    molecule_loader = MoleculeLoader(molecule_path)
    molecule = molecule_loader.molecules[0]

    repres = Representatives()
    pca = repres._get_pca(molecule.df)

    datatypes = {
        "atom_id": int,
        "atom_name": object,
        "res_id": object,
        "res_name": object,
        "subst_name": object,
        "x": float,
        "y": float,
        "z": float,
        "charge": float,
        "pc_type": object,
        "pc_id": object,
        "pc_atom_id": object,
    }

    for index, datatype in pca.dtypes.items():
        assert datatype == datatypes[index]


@pytest.mark.parametrize("filename", [("AAK1_4wsq_altA_chainA_reduced.mol2")])
def test_get_pca_pc_datatypes(filename):

    # Load molecule
    molecule_path = Path(__name__).parent / "ratar" / "tests" / "data" / filename
    molecule_loader = MoleculeLoader(molecule_path)
    molecule = molecule_loader.molecules[0]

    repres = Representatives()
    pc = repres._get_pc(molecule.df)

    datatypes = {
        "atom_id": object,
        "atom_name": object,
        "res_id": object,
        "res_name": object,
        "subst_name": object,
        "x": float,
        "y": float,
        "z": float,
        "charge": float,
        "pc_type": object,
        "pc_id": object,
        "pc_atom_id": object,
    }

    for index, datatype in pc.dtypes.items():
        assert datatype == datatypes[index]
