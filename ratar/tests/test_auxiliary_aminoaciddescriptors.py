"""
Unit and regression test for the AminoAcidDescriptor class in the ratar.auxiliary module of the ratar package.
"""

import sys

import pytest
from pathlib import Path

from ratar.auxiliary import MoleculeLoader, AminoAcidDescriptors


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
    molecule_path = Path(sys.path[0]) / 'ratar' / 'tests' / 'data' / filename
    molecule_loader = MoleculeLoader()
    molecule_loader.load_molecule(molecule_path)
    molecule = molecule_loader.pmols[0].df

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
