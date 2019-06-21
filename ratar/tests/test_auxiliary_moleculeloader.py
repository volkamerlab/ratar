"""
Unit and regression test for the MoleculeLoader class in the ratar.auxiliary module of the ratar package.
"""

import sys

import pytest
from pathlib import Path

from ratar.auxiliary import MoleculeLoader


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
    molecule_path = Path(sys.path[0]) / 'ratar' / 'tests' / 'data' / filename
    molecule_loader = MoleculeLoader()
    molecule_loader.load_molecule(molecule_path)

    assert len(molecule_loader.pmols) == len(code)

    for c, v in enumerate(molecule_loader.pmols):
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
    molecule_path = Path(sys.path[0]) / 'ratar' / 'tests' / 'data' / filename
    molecule_loader = MoleculeLoader()
    molecule_loader.load_molecule(molecule_path, remove_solvent=True)

    assert molecule_loader.pmols[0].df.shape == (n_atoms, 9)
