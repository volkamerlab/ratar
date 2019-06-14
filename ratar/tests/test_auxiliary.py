"""
Unit and regression test for the ratar.auxiliary module of the ratar package.
"""

import sys

import pytest
from pathlib import Path

from ratar.auxiliary import MoleculeLoader, AminoAcidDescriptors, load_pseudocenters


@pytest.mark.parametrize('filename, code, n_atoms, centroid', [
    (
        'scpdb_1a9p1.mol2',
        ['1a9p_9DI_1_site'],
        [463],
        [[22.346283, 90.841438, 74.726662]]
    ),
    (
        'AAK1_4wsq_altA_chainA.mol2',
        ['HUMAN/AAK1_4wsq_altA_chainA'],
        [1326],
        [[1.41567, 20.950211, 36.020875]]
    ),
    (
        'scpdb_egfr_20190128.mol2',
        ['1m17_AQ4_1_site', '1xkk_FMM_1_site'],
        [446, 714],
        [[25.649973, -1.22486, 54.142726], [15.226475, 34.955579, 39.749625]]
     ),
    (
        'toughm1_1a0iA.pdb',
        ['data_toughm1_1a0iA'],
        [160],
        [[5.046231, -19.593294, 59.772319]]
    )
])
def test_molecule_loader(filename, code, n_atoms, centroid):
    """
    Test MoleculeLoader class.

    Parameters
    ---------
    filename : str
        Name of molecule file.
    code : str
        Name of molecule code.
    n_atoms : int
        Number of atoms (i.e. number of DataFrame rows).
    centroid : list of lists
        Centroid(s) of molecule.
    """

    # Load molecule
    path = Path(sys.path[0]) / 'ratar' / 'tests' / 'data' / filename
    mol_loader = MoleculeLoader(path)
    mol_loader.load_molecule()

    assert len(mol_loader.pmols) == len(code)

    for c, v in enumerate(mol_loader.pmols):
        assert v.code == code[c]
        assert v.df.shape == (n_atoms[c], 9)
        assert list(v.df.columns) == ['atom_id', 'atom_name', 'res_id', 'res_name', 'subst_name', 'x', 'y', 'z', 'charge']
        assert abs(v.df['x'].mean() - centroid[c][0]) < 0.0001
        assert abs(v.df['y'].mean() - centroid[c][1]) < 0.0001
        assert abs(v.df['z'].mean() - centroid[c][2]) < 0.0001


@pytest.mark.parametrize('filename, n_atoms', [
    (
        'scpdb_1a9p1.mol2',
        457,
    )
])
def test_molecule_loader_remove_solvent(filename, n_atoms):
    """
    Test MoleculeLoader class.

    Parameters
    ---------
    filename : str
        Name of molecule file.
    n_atoms : int
        Number of atoms (i.e. number of DataFrame rows).
    """

    # Load molecule
    path = Path(sys.path[0]) / 'ratar' / 'tests' / 'data' / filename
    mol_loader = MoleculeLoader(path)
    mol_loader.load_molecule(remove_solvent=True)

    assert mol_loader.pmols[0].df.shape == (n_atoms, 9)


@pytest.mark.parametrize('filename, n_atoms, centroid', [
    (
        'scpdb_1a9p1.mol2',
        457,
        [22.38694, 90.907972, 74.754425]
    )
])
def test_amino_acid_descriptors(filename, n_atoms, centroid):
    """
    Test AminoAcidDescriptor class.

    Parameters
    ----------
    filename : str
        Name of molecule file.
    n_atoms : int
        Number of atoms (i.e. number of DataFrame rows)
    centroid : list of lists
        Centroid of molecule.
    """

    # Load molecule
    path = Path(sys.path[0]) / 'ratar' / 'tests' / 'data' / filename
    mol_loader = MoleculeLoader(path)
    mol_loader.load_molecule()
    molecule = mol_loader.pmols[0].df

    # Load amino acid descriptors
    aa_descriptors = AminoAcidDescriptors()

    # Get molecule atoms that have Z-scales information available
    molecule_aa_zscales = aa_descriptors.get_zscales_amino_acids(molecule)

    assert aa_descriptors.zscales.shape == (20, 5)
    assert abs(aa_descriptors.zscales.mean().mean() - 0.028100) < 0.0001

    assert molecule_aa_zscales.shape == (n_atoms, 9)
    assert abs(molecule_aa_zscales['x'].mean() - centroid[0]) < 0.0001
    assert abs(molecule_aa_zscales['y'].mean() - centroid[1]) < 0.0001
    assert abs(molecule_aa_zscales['z'].mean() - centroid[2]) < 0.0001


@pytest.mark.parametrize('filename, n_atoms, centroid', [
    (
        'scpdb_1a9p1.mol2',
        457,
        [22.38694, 90.907972, 74.754425]
    )
])
def test_load_pseudocenters():

    pc = load_pseudocenters()

    assert pc.shape == (71, 4)
    assert pc.loc[0]['pc_atom_id'] == 'ALA_H_1_CB'
    assert pc.loc[0]['pc_id'] == 'ALA_H_1'
    assert pc.loc[0]['pattern'] == 'ALA_CB'
    assert pc.loc[0]['type'] == 'H'
