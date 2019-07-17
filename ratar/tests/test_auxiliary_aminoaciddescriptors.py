"""
Unit and regression test for the AminoAcidDescriptor class in the ratar.auxiliary module of the ratar package.
"""

from pathlib import Path

import numpy as np
import pytest

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
    molecule_path = Path(__name__).parent / 'ratar' / 'tests' / 'data' / filename
    molecule_loader = MoleculeLoader(molecule_path)
    molecule = molecule_loader.molecules[0]

    # Load amino acid descriptors
    aa_descriptors = AminoAcidDescriptors()

    # Get molecule atoms that have Z-scales information available
    molecule_aa_zscales = aa_descriptors.get_zscales_amino_acids(molecule.df)

    assert aa_descriptors.zscales.shape == (20, 5)
    assert np.isclose(aa_descriptors.zscales.mean().mean(), 0.028100)

    assert molecule_aa_zscales.shape == (n_atoms, 9)
    assert np.isclose(molecule_aa_zscales['x'].mean(), centroid[0], rtol=1e-04)
    assert np.isclose(molecule_aa_zscales['y'].mean(), centroid[1], rtol=1e-04)
    assert np.isclose(molecule_aa_zscales['z'].mean(), centroid[2], rtol=1e-04)
