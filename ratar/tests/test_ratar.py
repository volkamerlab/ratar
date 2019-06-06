"""
Unit and regression test for the ratar package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest
from pathlib import Path
import pickle

from ratar.auxiliary import MoleculeLoader
from ratar.encoding import BindingSite


def test_ratar_imported():
    """
    Sample test, will always pass so long as import statement worked
    """
    assert 'ratar' in sys.modules


@pytest.mark.parametrize('filename, mol_code, mol_df_n_cols, mol_df_n_rows', [
    ('scpdb_1a9p1.mol2', '1a9p_9DI_1_site', 463, 9),
    ('toughm1_1a0iA.pdb', 'data_toughm1_1a0iA', 160, 9)
])
def test_molfileloader(filename, mol_code, mol_df_n_rows, mol_df_n_cols):
    """
    Test if molecule structure file is loaded correctly.

    Parameters
    ---------
    filename : str
        Name of molecule file.
    mol_code : str
        Name of molecule code.
    mol_df_n_rows : int
        Number of DataFrame rows.
    mol_df_n_cols: int
        Number of DataFrame columns.
    """

    path = Path(sys.path[0]) / 'ratar' / 'tests' / 'data' / filename
    mol_loader = MoleculeLoader(path)

    assert len(mol_loader.pmols) == 1
    assert mol_loader.pmols[0].code == mol_code
    assert mol_loader.pmols[0].df.shape[0] == mol_df_n_cols
    assert mol_loader.pmols[0].df.shape[1] == mol_df_n_rows
    # TODO test centroid of structure


@pytest.mark.parametrize('mol_file1, mol_file2', [
    ('AAK1_4wsq_altA_chainA.mol2', 'AAK1_4wsq_altA_chainB.mol2')
])
def test_bindingsites_eq(mol_file1, mol_file2):
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
def test_bindingsites(mol_file, encoding_file):
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
