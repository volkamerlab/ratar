"""
Unit and regression test for the Representatives class in the ratar.encoding module of the ratar package.
"""

import sys

import pytest
from pathlib import Path

from ratar.auxiliary import MoleculeLoader
from ratar.encoding import Representatives


@pytest.mark.parametrize('filename, mol_n_atoms, ca_n_atoms', [
    (
        'AAK1_4wsq_altA_chainA_reduced.mol2',
        94,
        8
    )
])
def test_get_ca(filename, mol_n_atoms, ca_n_atoms):
    """
    Test if pseudocenter atoms are correctly extracted from molecule.

    Parameters
    ----------
    filename : str
        Name of molecule file.
    mol_n_atoms : int
        Number of atoms in molecule.
    ca_n_atoms : int
        Number of Calpha atoms in molecule.
    """

    # Load molecule
    molecule_path = Path(sys.path[0]) / 'ratar' / 'tests' / 'data' / filename
    molecule_loader = MoleculeLoader()
    molecule_loader.load_molecule(molecule_path)
    molecule = molecule_loader.pmols[0].df

    # Set representatives
    representatives = Representatives()
    representatives_pca = representatives._get_ca(molecule)

    assert molecule.shape == (mol_n_atoms, 9)
    assert representatives_pca.shape == (ca_n_atoms, 9)


@pytest.mark.parametrize('filename, mol_n_atoms, pca_n_atoms,', [
    (
        'AAK1_4wsq_altA_chainA_reduced.mol2',
        94,
        34
    )
])
def test_get_pca(filename, mol_n_atoms, pca_n_atoms):
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
    """

    # Load molecule
    molecule_path = Path(sys.path[0]) / 'ratar' / 'tests' / 'data' / filename
    molecule_loader = MoleculeLoader()
    molecule_loader.load_molecule(molecule_path)
    molecule = molecule_loader.pmols[0].df

    # Set representatives
    representatives = Representatives()
    representatives_pca = representatives._get_pca(molecule)

    assert molecule.shape == (mol_n_atoms, 9)
    assert representatives_pca.shape == (pca_n_atoms, 12)


@pytest.mark.parametrize('filename, mol_n_atoms, pc_n_atoms', [
    (
        'AAK1_4wsq_altA_chainA_reduced.mol2',
        94,
        29
    )
])
def test_get_pc(filename, mol_n_atoms, pc_n_atoms):
    """
    Test if pseudocenter atoms are correctly extracted from molecule.

    Parameters
    ----------
    filename : str
        Name of molecule file.
    mol_n_atoms : int
        Number of atoms in molecule.
    pc_n_atoms : int
        Number of pseudocenters in molecule.
    """

    # Load molecule
    molecule_path = Path(sys.path[0]) / 'ratar' / 'tests' / 'data' / filename
    molecule_loader = MoleculeLoader()
    molecule_loader.load_molecule(molecule_path)
    molecule = molecule_loader.pmols[0].df

    # Set representatives
    representatives = Representatives()
    representatives_pc = representatives._get_pc(molecule)

    assert molecule.shape == (mol_n_atoms, 9)
    assert representatives_pc.shape == (pc_n_atoms, 12)

