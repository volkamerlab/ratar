"""
Unit and regression test for the ratar.encoding module of the ratar package.
"""

import sys

import pytest
from pathlib import Path
import pickle

from ratar.auxiliary import MoleculeLoader
from ratar.encoding import BindingSite, Representatives


@pytest.mark.parametrize('filename, mol_n_atoms, pca_n_atoms, pc_n_atoms', [
    ('AAK1_4wsq_altA_chainA_reduced.mol2', 94, 34, 29)
])
def test_get_pseudocenter_data(filename, mol_n_atoms, pca_n_atoms, pc_n_atoms):
    """
    Test if pseudocenter atoms are correctly extracted from molecule.

    Parameters
    ----------
    filename : str
        Name of molecule file.
    mol_n_atoms : int
        Number of atoms in molecule.
    pca_n_atoms : int
        Number of pseudocenter atoms in molecule.
    pc_n_atoms : int
        Number of pseudocenters in molecule.
    """

    # Load molecule
    path = Path(sys.path[0]) / 'ratar' / 'tests' / 'data' / filename
    mol_loader = MoleculeLoader(path)
    mol_loader.load_molecule()
    molecule = mol_loader.pmols[0].df

    # Set representatives
    representatives = Representatives('')
    representatives_pca = representatives._get_pca(molecule)
    representatives_pc = representatives._get_pc(molecule)

    assert molecule.shape == (mol_n_atoms, 9)
    assert representatives_pca.shape == (pca_n_atoms, 12)
    assert representatives_pc.shape == (pc_n_atoms, 12)


@pytest.mark.parametrize('mol_file1, mol_file2', [
    ('AAK1_4wsq_altA_chainA.mol2', 'AAK1_4wsq_altA_chainB.mol2')
])
def ttest_bindingsites_eq(mol_file1, mol_file2):
    """
    Test __eq__ functions for encoding classes.

    Parameters
    ----------
    mol_file1 : str
        Name of file containing the structure for molecule A.
    mol_file2 : str
        Name of file containing the structure for molecule B.

    """

    path1 = Path(sys.path[0]) / 'ratar' / 'tests' / 'data' / mol_file1
    path2 = Path(sys.path[0]) / 'ratar' / 'tests' / 'data' / mol_file2

    pmols1 = MoleculeLoader(path1)
    pmols2 = MoleculeLoader(path2)

    bs1 = BindingSite(pmols1.pmols[0])
    bs2 = BindingSite(pmols1.pmols[0])
    bs3 = BindingSite(pmols2.pmols[0])

    assert (bs1 == bs2) is True
    assert (bs1 == bs3) is False


@pytest.mark.parametrize('mol_file, encoding_file', [
    ('AAK1_4wsq_altA_chainA.mol2', 'AAK1_4wsq_altA_chainA_encoded.p')
])
def ttest_bindingsites(mol_file, encoding_file):
    """
    Test if ratar-encoding of a molecule (based on its structural file) produces the same result
    as the reference encoding for the same molecule.

    Parameters
    ----------
    mol_file : str
        Name of file containing the structure of molecule A.
    encoding_file : str
        Name of file containing ratar-encoding for molecule A.
    """

    # Encode binding site
    path_mol = Path(sys.path[0]) / 'ratar' / 'tests' / 'data' / mol_file
    pmols = MoleculeLoader(path_mol)
    bs = BindingSite(pmols.pmols[0])

    # Load reference binding site encoding
    path_encoding = Path(sys.path[0]) / 'ratar' / 'tests' / 'data' / encoding_file
    with open(path_encoding, 'rb') as f:
        bs_ref = pickle.load(f)

    # Compare encoding with reference encoding
    assert (bs == bs_ref) is True
