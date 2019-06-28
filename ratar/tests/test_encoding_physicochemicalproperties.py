"""
Unit and regression test for the PhysicoChemicalProperties class in the ratar.encoding module of the ratar package.
"""

from flatten_dict import flatten
import numpy as np
import pytest
from pathlib import Path

from ratar.auxiliary import MoleculeLoader
from ratar.encoding import PhysicoChemicalProperties


@pytest.mark.parametrize('mol_file1, mol_file2', [
    ('AAK1_4wsq_altA_chainA.mol2', 'AAK1_4wsq_altA_chainB.mol2')
])
def test_physicochemicalproperties_eq(mol_file1, mol_file2):
    """
    Test __eq__ function for PhysicoChemicalProperties class.

    Parameters
    ----------
    mol_file1 : str
        Name of file containing the structure for molecule A.
    mol_file2 : str
        Name of file containing the structure for molecule B.

    """

    molecule_path1 = Path(__name__).parent / 'ratar' / 'tests' / 'data' / mol_file1
    molecule_path2 = Path(__name__).parent / 'ratar' / 'tests' / 'data' / mol_file2

    molecule_loader1 = MoleculeLoader()
    molecule_loader2 = MoleculeLoader()

    molecule_loader1.load_molecule(molecule_path1)
    molecule_loader2.load_molecule(molecule_path2)

    physicochemicalproperties1 = PhysicoChemicalProperties()
    physicochemicalproperties2 = PhysicoChemicalProperties()
    physicochemicalproperties3 = PhysicoChemicalProperties()

    physicochemicalproperties1.get_physicochemicalproperties_from_molecule(molecule_loader1.get_first_molecule())
    physicochemicalproperties2.get_physicochemicalproperties_from_molecule(molecule_loader1.get_first_molecule())
    physicochemicalproperties3.get_physicochemicalproperties_from_molecule(molecule_loader2.get_first_molecule())

    assert (physicochemicalproperties1 == physicochemicalproperties2)
    assert not (physicochemicalproperties1 == physicochemicalproperties3)


@pytest.mark.parametrize('filename, column_names, n_atoms, centroid', [
    (
        'AAK1_4wsq_altA_chainA_reduced.mol2',
        dict(
            zip(
                'ca/z1 ca/z123 pca/z1 pca/z123 pc/z1 pc/z123'.split(),
                [
                    ['z1'],
                    ['z1', 'z2', 'z3'],
                    ['z1'],
                    ['z1', 'z2', 'z3'],
                    ['z1'],
                    ['z1', 'z2', 'z3']]
            )
        ),
        dict(
            zip(
                'ca/z1 ca/z123 pca/z1 pca/z123 pc/z1 pc/z123'.split(),
                [8, 8, 34, 34, 29, 29]
            )
        ),
        dict(
            zip(
                'ca/z1 ca/z123 pca/z1 pca/z123 pc/z1 pc/z123'.split(),
                [
                    [-0.066250],
                    [-0.066250, -1.49, -0.108750],
                    [-0.588824],
                    [-0.588824, -1.490588, -0.376765],
                    [-0.068966],
                    [-0.068966, -1.431034, -0.181379]]
            )
        )
    )
])
def test_get_physicochemicalproperties_from_molecule(filename, column_names, n_atoms, centroid):
    """
    Test if coordinates are correctly extracted from representatives of a molecule.

    Parameters
    ----------
    filename : str
        Name of molecule file.
    column_names : list of str
        List of molecule DataFrame columns.
    n_atoms : dict of int
        Number of atoms for each representatives type.
    centroid : dict of list of float
        3D coordinates of molecule centroid for each representatives type.
    """

    # Load molecule
    molecule_path = Path(__name__).parent / 'ratar' / 'tests' / 'data' / filename
    molecule_loader = MoleculeLoader()
    molecule_loader.load_molecule(molecule_path)
    molecule = molecule_loader.get_first_molecule()

    # Set physicochemical properties
    physicochemicalproperties = PhysicoChemicalProperties()
    physicochemicalproperties.get_physicochemicalproperties_from_molecule(molecule)

    pcp_flat = flatten(physicochemicalproperties.data, reducer='path')

    for key, value in pcp_flat.items():
        assert all(value.columns == column_names[key])
        assert value.shape[0] == n_atoms[key]

        for i, column_name in enumerate(value.columns):
            print(value[column_name].mean())
            print(centroid[key][i])
            assert np.isclose(value[column_name].mean(), centroid[key][i])
