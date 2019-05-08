"""
auxiliary.py

Read-across the targetome -
An integrated structure- and ligand-based workbench for computational target prediction and novel tool compound design

Handles the helper functions.

"""


########################################################################################
# Import modules
########################################################################################

import os
from pathlib import Path

from biopandas.mol2 import PandasMol2, split_multimol2
from biopandas.pdb import PandasPdb
import pandas as pd


########################################################################################
# Global variables
########################################################################################

# Project location
package_path = Path('/home/dominique/Documents/projects/ratar/ratar')


########################################################################################
# Load auxiliary data
########################################################################################

class AminoAcidDescriptors:

    """
    Class used to store amino acid descriptor data, e.g. Z-scales.

    Attributes
    ----------
    zscales : DataFrame
        Z-scales for standard and a few non-standard amino acids.

    """

    def __init__(self):
        # Z-scales taken from: https://github.com/Superzchen/iFeature/blob/master/codes/ZSCALE.py
        zscales_path = package_path / 'data/zscales.csv'
        self.zscales = pd.read_csv(str(zscales_path), index_col='aa3')


########################################################################################
# IO functions
########################################################################################

class MolFileLoader:

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
    pmols : list of BioPandas objects
        List of molecule data in the form of BioPandas objects.

    """

    def __init__(self, input_path):

        self.input_path = Path(input_path)
        self.pmols = None

        if self.input_path.suffix == '.pdb':
            self.pmols = self._load_pdb()
        elif self.input_path.suffix == '.mol2':
            self.pmols = self._load_mol2()

    def _load_mol2(self):

        """
        Load molecule data from a mol2 file, which can contain multiple entries.

        Returns
        -------
        List of BioPandas objects
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

            # Select columns of interest
            pmol._df = pmol.df.loc[:, ['atom_id',
                                       'atom_name',
                                       'subst_name',
                                       'x',
                                       'y',
                                       'z',
                                       'charge']]

            # Insert additional columns (split ASN22 to ASN and 22)
            pmol.df.insert(loc=2, column='res_id', value=[i[3:] for i in pmol.df['subst_name']])
            pmol.df.insert(loc=3, column='res_name', value=[i[:3] for i in pmol.df['subst_name']])

            pmols.append(pmol)

        return pmols

    def _load_pdb(self):

        """
        Load molecule data from a pdb file, which can contain multiple entries.

        Parameters
        ----------
        self.input_path : str
            Absolute path to pdb file.

        Returns
        -------
        List of BioPandas objects
            List of BioPandas objects containing metadata and structural data of molecule(s) in pdb file.

        """

        pmol = PandasPdb().read_pdb(str(self.input_path))

        # If object has no code, set string from file stem and its folder name
        if pmol.code == "":
            pmol.code = '_'.join([self.input_path.parts[-2], self.input_path.stem])

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

        # Cast to list only for homogeneity with reading mol2 files that can have multiple entries
        pmols = [pmol]

        return pmols


def create_directory(directory):

    """
    Create directory if it does not exist.

    Parameters
    ----------
    directory : str
        Absolute path to directory.

    """

    # Cast to Path
    directory = Path(directory)

    try:
        if not directory.exists():
            directory.mkdir(parents=True)
    except OSError:
        print(f'OSError: Creating directory {directory} failed.')
