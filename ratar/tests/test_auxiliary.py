"""
Unit and regression test for functions in the ratar.auxiliary module of the ratar package.
"""

import pytest

from ratar.auxiliary import load_pseudocenters


@pytest.mark.parametrize(
    "df_index, pc_atom_id, pc_atom_pattern, pc_id, pc_type",
    [(7, "ASN_HBA_1_OD1", "ASN_OD1", "ASN_HBA_1", "HBA")],
)
def test_load_pseudocenters(df_index, pc_atom_id, pc_atom_pattern, pc_id, pc_type):

    pc = load_pseudocenters(remove_hbda=False)

    assert pc.shape == (76, 4)
    assert pc.loc[df_index]["pc_atom_id"] == pc_atom_id
    assert pc.loc[df_index]["pc_atom_pattern"] == pc_atom_pattern
    assert pc.loc[df_index]["pc_id"] == pc_id
    assert pc.loc[df_index]["pc_type"] == pc_type


@pytest.mark.parametrize(
    "df_index, pc_atom_id, pc_atom_pattern, pc_id, pc_type",
    [(7, "ASN_HBA_1_OD1", "ASN_OD1", "ASN_HBA_1", "HBA")],
)
def test_load_pseudocenters_remove_hbda(df_index, pc_atom_id, pc_atom_pattern, pc_id, pc_type):

    pc = load_pseudocenters(remove_hbda=True)

    assert pc.shape == (71, 4)
    assert pc.loc[df_index]["pc_atom_id"] == pc_atom_id
    assert pc.loc[df_index]["pc_atom_pattern"] == pc_atom_pattern
    assert pc.loc[df_index]["pc_id"] == pc_id
    assert pc.loc[df_index]["pc_type"] == pc_type
