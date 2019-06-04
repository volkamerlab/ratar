"""
Unit and regression test for the ratar package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest
from pathlib import Path

from ratar.auxiliary import MolFileLoader


def test_ratar_imported():
    """
    Sample test, will always pass so long as import statement worked
    """
    assert "ratar" in sys.modules


@pytest.mark.parametrize('path, mol_code, mol_df_n_cols, mol_df_n_rows', [
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
    assert mol_loader.pmols[0].df.shape[0] == mol_df_n_rows
    assert mol_loader.pmols[0].df.shape[1] == mol_df_n_cols






