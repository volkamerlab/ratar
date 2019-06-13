"""
auxiliary.py

Read-across the targetome -
An integrated structure- and ligand-based workbench for computational target prediction and novel tool compound design

Handles the helper functions.

"""

import logging

from pathlib import Path
from biopandas.mol2 import PandasMol2, split_multimol2
from biopandas.pdb import PandasPdb
import pandas as pd
import pickle

import ratar

logger = logging.getLogger(__name__)


# Project location
ratar_path = Path(ratar.__file__).parent


class AminoAcidDescriptors:
    """
    Class used to store amino acid descriptor data, e.g. Z-scales.

    Attributes
    ----------
    zscales : pandas.DataFrame
        Z-scales for standard and a few non-standard amino acids.

    Notes
    -----
    Z-scales taken from: https://github.com/Superzchen/iFeature/blob/master/codes/ZSCALE.py
    """

    def __init__(self):
        zscales_path = ratar_path / 'data' / 'zscales.csv'
        self.zscales = pd.read_csv(str(zscales_path), index_col='aa3')

    def _get_zscales_amino_acids(self, molecule, output_log_path=None):
        """
        Get all amino acids atoms that are described by Z-scales.

        Parameters
        ----------
        molecule : pandas.DataFrame
            DataFrame containing atom lines from input file.
        output_log_path : str
            Path to log file.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing atom lines from input file described by Z-scales.
        """

        # Get amino acid name per row (atom)
        mol_aa = molecule['res_name']

        # Get only rows (atoms) that belong to Z-scales amino acids
        mol_zscales_aa = molecule[mol_aa.apply(lambda y: y in self.zscales.index)].copy()

        # Get only rows (atoms) that DO NOT belong to Z-scales amino acids
        mol_non_zscales_aa = molecule[mol_aa.apply(lambda y: y not in self.zscales.index)].copy()

        if not mol_non_zscales_aa.empty:
            if output_log_path is not None:
                with open(output_log_path, 'a+') as f:
                    f.write('Atoms removed for binding site encoding:\n\n')
                    f.write(mol_non_zscales_aa.to_string() + '\n\n')
            else:
                print('Atoms removed for binding site encoding:')
                print(mol_non_zscales_aa)

        return mol_zscales_aa


class MoleculeLoader:
    """
    Class used to load molecule data from mol2 and pdb files in the form of unified BioPandas objects.

    Parameters
    ----------
    input_path : str
        Absolute path to a mol2 (can contain multiple entries) or pdb file.

    Attributes
    ----------
    input_path : pathlib.PosixPath
        Absolute path to a mol2 (can contain multiple entries) or pdb file.
    pmols : list of biopandas.mol2.pandas_mol2.PandasMol2 or biopandas.pdb.pandas_pdb.PandasPdb
        List of molecule data in the form of BioPandas objects.
    """

    def __init__(self, input_path):

        self.input_path = Path(input_path)
        self.pmols = None

    def load_molecule(self, remove_solvent=False):

        if self.input_path.exists():
            logger.info(f'File to be loaded: {self.input_path}', extra={'molecule_id': 'all'})
        else:
            logger.error(f'File not found: {self.input_path}', extra={'molecule_id': 'all'})
            raise FileNotFoundError(f'File not found: {self.input_path}')

        if self.input_path.suffix == '.pdb':
            self.pmols = self._load_pdb(remove_solvent)
        elif self.input_path.suffix == '.mol2':
            self.pmols = self._load_mol2(remove_solvent)

        logger.info('File loaded.', extra={'molecule_id': 'all'})

        return self.pmols

    def _load_mol2(self, remove_solvent=False):
        """
        Load molecule data from a mol2 file, which can contain multiple entries.

        Returns
        -------
        list of biopandas.mol2.pandas_mol2.PandasMol2
            List of BioPandas objects containing metadata and structural data of molecule(s) in mol2 file.
        """

        # In case of multiple entries in one mol2 file, include iteration step
        pmols = []

        for mol2 in split_multimol2(str(self.input_path)):

            # Mol2 files can have 9 or 10 columns.
            try:  # Try 9 columns.
                pmol = PandasMol2().read_mol2_from_list(
                                                        mol2_code=mol2[0],
                                                        mol2_lines=mol2[1],
                                                        columns={0: ('atom_id', int),
                                                                 1: ('atom_name', str),
                                                                 2: ('x', float),
                                                                 3: ('y', float),
                                                                 4: ('z', float),
                                                                 5: ('atom_type', str),
                                                                 6: ('subst_id', str),
                                                                 7: ('subst_name', str),
                                                                 8: ('charge', float)}
                                                        )

            except AssertionError:  # If 9 columns did not work, try 10 columns.
                pmol = PandasMol2().read_mol2_from_list(
                                                        mol2_code=mol2[0],
                                                        mol2_lines=mol2[1],
                                                        columns={0: ('atom_id', int),
                                                                 1: ('atom_name', str),
                                                                 2: ('x', float),
                                                                 3: ('y', float),
                                                                 4: ('z', float),
                                                                 5: ('atom_type', str),
                                                                 6: ('subst_id', str),
                                                                 7: ('subst_name', str),
                                                                 8: ('charge', float),
                                                                 9: ('status_bit', str)}
                                                        )

            # Remove all '/' from code (code used as folder name, '/' would cause subdirectory creation)
            pmol.code = pmol.code.replace('/', '_')

            # Insert additional columns (split ASN22 to ASN and 22)
            res_id_list = []
            res_name_list = []

            for i, j in zip(pmol.df['subst_name'], pmol.df['atom_type']):
                if i[:2] == j.upper():
                    # These are ions such as CA or MG
                    res_id_list.append(i[2:])
                    res_name_list.append(i[:2])
                else:
                    # These are amino acid, linkers, compounds, ...
                    res_id_list.append(i[3:])
                    res_name_list.append(i[:3])

            pmol.df.insert(loc=2, column='res_id', value=res_id_list)
            pmol.df.insert(loc=2, column='res_name', value=res_name_list)

            # Select columns of interest
            pmol._df = pmol.df.loc[:, ['atom_id',
                                       'atom_name',
                                       'res_id',
                                       'res_name',
                                       'subst_name',
                                       'x',
                                       'y',
                                       'z',
                                       'charge']]

            # Remove solvent if parameter remove_solvent=True
            if remove_solvent:
                ix = pmol.df.index[pmol.df['res_name'] == 'HOH']
                pmol.df.drop(index=ix, inplace=True)

            pmols.append(pmol)

        return pmols

    def _load_pdb(self, remove_solvent=False):
        """
        Load molecule data from a pdb file, which can contain multiple entries.

        Parameters
        ----------
        self.input_path : str
            Absolute path to pdb file.

        Returns
        -------
        list of biopandas.pdb.pandas_pdb.PandasPdb
            List of BioPandas objects containing metadata and structural data of molecule(s) in pdb file.
        """

        pmol = PandasPdb().read_pdb(str(self.input_path))

        # If object has no code, set string from file stem and its folder name
        if pmol.code == "":
            pmol.code = '_'.join([self.input_path.parts[-2], self.input_path.stem]).replace('/', '_')

        # Get both ATOM and HETATM lines of PDB file
        pmol._df = pd.concat([pmol.df['ATOM'], pmol.df['HETATM']])

        # Select columns of interest
        pmol._df = pmol.df.loc[:, ['atom_number',
                                   'atom_name',
                                   'residue_number',
                                   'residue_name',
                                   'x_coord',
                                   'y_coord',
                                   'z_coord',
                                   'charge']]

        # Insert additional columns
        pmol.df.insert(loc=4,
                       column='subst_name',
                       value=[f'{i}{j}' for i, j in zip(pmol.df['residue_name'], pmol.df['residue_number'])])

        # Rename columns
        pmol.df.rename(index=str, inplace=True, columns={'atom_number': 'atom_id',
                                                         'residue_number': 'res_id',
                                                         'residue_name': 'res_name',
                                                         'x_coord': 'x',
                                                         'y_coord': 'y',
                                                         'z_coord': 'z'})

        # Remove solvent if parameter remove_solvent=True
        if remove_solvent:
            ix = pmol.df.index[pmol.df['res_name'] == 'HOH']
            pmol.df.drop(index=ix, inplace=True)

        # Cast to list only for homogeneity with reading mol2 files that can have multiple entries
        pmols = [pmol]

        return pmols


def load_pseudocenters():
    """
    Load pseudocenters from file.
    Remove HBDA features, since they contain too few data points for encoding.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing pseudocenter information.
    """
    with open(ratar_path / 'data' / 'pseudocenter_atoms.p', 'rb') as f:
        pc_atoms = pickle.load(f)

    # Remove HBDA features information (too few data points)
    pc_atoms = pc_atoms[pc_atoms['type'] != 'HBDA']
    pc_atoms.reset_index(drop=True, inplace=True)

    return pc_atoms


def create_directory(directory):
    """
    Create directory if it does not exist.

    Parameters
    ----------
    directory : str or pathlib.Path
        Absolute path to directory.
    """

    # Cast to Path
    directory = Path(directory)

    try:
        if not directory.exists():
            directory.mkdir(parents=True)
    except OSError:
        raise OSError(f'Creating directory failed: {directory}')
