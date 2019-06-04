"""
Unit and regression test for the ratar package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest
from pathlib import Path

from ratar.auxiliary import MolFileLoader
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
    mol_loader = MolFileLoader(path)

    assert len(mol_loader.pmols) == 1
    assert mol_loader.pmols[0].code == mol_code
    assert mol_loader.pmols[0].df.shape[0] == mol_df_n_cols
    assert mol_loader.pmols[0].df.shape[1] == mol_df_n_rows


@pytest.mark.parametrize('filename1, filename2', [
    ('AAK1_4wsq_altA_chainA.mol2', 'AAK1_4wsq_altA_chainB.mol2')
])
def test_equal_bindingsites(filename1, filename2):
    """
    Test __eq__ functions for encoding classes.

    Parameters
    ----------
    filename1 : str
        Name of molecule file 1.
    filename2 : str
        Name of molecule file 2.

    """

    path1 = Path(sys.path[0]) / 'ratar' / 'tests' / 'data' / filename1
    path2 = Path(sys.path[0]) / 'ratar' / 'tests' / 'data' / filename2

    pmols1 = MolFileLoader(path1)
    pmols2 = MolFileLoader(path2)

    bs1 = BindingSite(pmols1.pmols[0])
    bs2 = BindingSite(pmols1.pmols[0])
    bs3 = BindingSite(pmols2.pmols[0])

    assert (bs1 == bs2) is True
    assert (bs1 == bs3) is False

