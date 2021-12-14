"""
Unit and regression test for functions the ratar.encoding module of the ratar package.
"""

from pathlib import Path

import pytest

from ratar.auxiliary import MoleculeLoader
from ratar.encoding import BindingSite, save_binding_site, save_cgo_file, process_encoding


@pytest.mark.parametrize('mol_file', [
    'AAK1_4wsq_altA_chainA.mol2'
])
def test_save_binding_site(mol_file):
    """
    Test whether the binding site pickle file is saved when calling the ratar.encoding.save_binding_site function.

    Parameters
    ----------
    mol_file : str
        Name of file containing the structure for molecule.
    """

    molecule_path = Path(__name__).parent / 'ratar' / 'tests' / 'data' / mol_file
    molecule_loader = MoleculeLoader(molecule_path)
    bindingsite = BindingSite(molecule_loader.molecules[0])

    output_path = Path(__name__).parent / 'ratar' / 'tests' / 'data' / 'tmp' / 'bindingsite.p'

    # Remove this file if exists already
    if output_path.exists():
        output_path.unlink()

    # Save file
    save_binding_site(bindingsite, output_path)

    # Check if file exists
    assert output_path.exists()

    # Remove file if exists
    if output_path.exists():
        output_path.unlink()


@pytest.mark.parametrize('mol_file', [
    'AAK1_4wsq_altA_chainA.mol2'
])
def test_save_cgo_file(mol_file):
    """
    Test whether the binding site cgo file is saved when calling the ratar.encoding.save_cgo_file function.

    Parameters
    ----------
    mol_file : str
        Name of file containing the structure for molecule.
    """

    molecule_path = Path(__name__).parent / 'ratar' / 'tests' / 'data' / mol_file
    molecule_loader = MoleculeLoader(molecule_path)
    bindingsite = BindingSite(molecule_loader.molecules[0])

    output_path = Path(__name__).parent / 'ratar' / 'tests' / 'data' / 'tmp' / 'ref_points.cgo'

    # Remove this file if exists already
    if output_path.exists():
        output_path.unlink()

    # Save file
    save_cgo_file(bindingsite, output_path)

    # Check if file exists
    assert output_path.exists()

    # Remove file if exists
    if output_path.exists():
        output_path.unlink()


@pytest.mark.parametrize('mol_file, output_path_p, output_path_cgo', [
    (
        'AAK1_4wsq_altA_chainA.mol2',
        Path('.') / 'encoding' / 'HUMAN' / 'AAK1_4wsq_altA_chainA' / 'ratar_encoding.p',
        Path('.') / 'encoding' / 'HUMAN' / 'AAK1_4wsq_altA_chainA' / 'ref_points_cgo.py'
    )
])
def test_process_encoding(mol_file, output_path_p, output_path_cgo):
    """
    Test whether the binding site cgo file is saved when calling the ratar.encoding.process_encoding function.

    Parameters
    ----------
    mol_file : str
        Name of file containing the structure for molecule.
    """

    molecule_path = Path(__name__).parent / 'ratar' / 'tests' / 'data' / mol_file
    output_dir = Path(__name__).parent / 'ratar' / 'tests' / 'data' / 'tmp'

    # Remove this file if exists already
    if (output_dir / 'encoding').exists():
        dir_file_list = [i for i in (output_dir / 'encoding').glob('**/*')]
        dir_file_list.reverse()
        [i.rmdir() if i.is_dir() else i.unlink() for i in dir_file_list]
        (output_dir / 'encoding').rmdir()

    # Save file
    process_encoding(molecule_path, output_dir)

    # Check if file exists
    assert (output_dir / output_path_p).exists()
    assert (output_dir / output_path_cgo).exists()

    # Remove this file if exists already
    if (output_dir / 'encoding').exists():
        dir_file_list = [i for i in (output_dir / 'encoding').glob('**/*')]
        dir_file_list.reverse()
        [i.rmdir() if i.is_dir() else i.unlink() for i in dir_file_list]
        (output_dir / 'encoding').rmdir()
