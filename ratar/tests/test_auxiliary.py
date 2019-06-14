"""
Unit and regression test for the ratar.auxiliary module of the ratar package.
"""

import pytest
from pathlib import Path

from ratar.auxiliary import MoleculeLoader


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