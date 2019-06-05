import sys

import pytest
from pathlib import Path
import pickle

from ratar.auxiliary import MolFileLoader
from ratar.encoding import BindingSite


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
    path_mol = Path('/home/dominique/Documents/projects/ratar/ratar/tests/data/') / mol_file
    pmols = MolFileLoader(path_mol)
    bs = BindingSite(pmols.pmols[0])

    # Load reference binding site encoding
    path_encoding = Path('/home/dominique/Documents/projects/ratar/ratar/tests/data/') / encoding_file
    with open(path_encoding, 'rb') as f:
        bs_ref = pickle.load(f)

    # Compare encoding with reference encoding
    print(bs == bs_ref)

    return bs


bs = test_bindingsites('AAK1_4wsq_altA_chainA.mol2', 'AAK1_4wsq_altA_chainA_encoded.p')
