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


class MoleculeLoader:
    """
    Class used to load molecule data from mol2 and pdb files in the form of unified BioPandas objects.

    Attributes
    ----------
    input_path : str or pathlib.PosixPath
        Absolute path to a mol2 (can contain multiple entries) or pdb file.
    pmols : list of biopandas.mol2.pandas_mol2.PandasMol2 or biopandas.pdb.pandas_pdb.PandasPdb
        List of molecule data in the form of BioPandas objects.
    n_molecules : int
        Number of molecules loaded.

    Example
    -------
    >>>> from ratar.auxiliary import MoleculeLoader

    >>>> molecule_path = '/path/to/pdb/or/mol2'
    >>>> molecule_loader = MoleculeLoader()
    >>>> molecule_loader.load_molecule(molecule_path, remove_solvent=True)

    >>>> pmols = molecule_loader.pmols
    >>>> molecule1 = pmols[0].df
    >>>> molecule1_id = pmols[0].code

    >>>> pmols[0].df == molecule_loader.get_first_molecule()
    True
    """

    def __init__(self):

        self.input_path = None
        self.pmols = None
        self.n_molecules = 0

    def load_molecule(self, input_path, remove_solvent=False):
        """
        Load one or multiple molecules from pdb or mol2 file.

        Parameters
        ----------
        input_path : str or pathlib.PosixPath
            Absolute path to a mol2 (can contain multiple entries) or pdb file.
        remove_solvent : bool
            Set True to remove solvent molecules (default: False).
        """

        # Set input path
        self.input_path = Path(input_path)

        if self.input_path.exists():
            logger.info(f'File to be loaded: {self.input_path}', extra={'molecule_id': 'all'})
        else:
            logger.error(f'File not found: {self.input_path}', extra={'molecule_id': 'all'})
            raise FileNotFoundError(f'File not found: {self.input_path}')

        # Load molecule data
        if self.input_path.suffix == '.pdb':
            self.pmols = self._load_pdb(remove_solvent)
        elif self.input_path.suffix == '.mol2':
            self.pmols = self._load_mol2(remove_solvent)

        # Set number of loaded molecules
        self.n_molecules = len(self.pmols)

        logger.info('File loaded.', extra={'molecule_id': 'all'})

        return None

    def get_first_molecule(self):
        """
        Convenience class method: get the first molecule DataFrame.

        Returns
        -------
        DataFrame
            Data for first molecule in MoleculeLoader class.
        """

        return self.pmols[0].df

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

    def get_zscales_amino_acids(self, molecule):
        """
        Get all amino acids atoms that are described by Z-scales.

        Parameters
        ----------
        molecule : pandas.DataFrame
            DataFrame containing atom lines from input file.

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
            logger.info(f'Atoms removed for binding site encoding: {mol_non_zscales_aa.to_string()}')

        return mol_zscales_aa


def load_pseudocenters():
    """
    Load pseudocenters from file.
    Remove HBDA features, since they contain too few data points for encoding.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing pseudocenter information.
    """

    pseudocenter_path = ratar_path / 'data' / 'pseudocenter_atoms.csv'

    pc_atoms = pd.read_csv(pseudocenter_path, index_col=0)

    # Remove HBDA features information (too few data points)
    # pc_atoms = pc_atoms[pc_atoms['pc_type'] != 'HBDA']
    # pc_atoms.reset_index(drop=True, inplace=True)

    return pc_atoms


def _preprocess_pseudocenters():
    """
    Preprocess pseudocenter csv file as a DataFrame containing per row one pseudocenter with columns for
    - pseudocenter ID, e.g. CYS_H_1,
    - residue name, e.g. CYS,
    - pseudocenter type, e.g. H, and
    - a list of origin atoms (atoms that belong to pseudocenter), e.g. ['CB', 'SG'].

    Returns
    -------
    pandas.DataFrame
        Pseudocenter data (pseudocenter ID, residue name, pseudocenter type and list of origin atoms).
    """

    # Load pseudocenter text file
    pc_df = pd.read_csv(ratar_path / "data" / "pseudocenters.csv", header=None)
    pc_df.columns = ["residue", "pc_type", "origin_atoms"]

    # Change case of amino acid column
    pc_df["residue"] = [i.upper() for i in pc_df["residue"]]

    # Cast column with multiple entry string to list
    pc_df["origin_atoms"] = [i.split(" ") for i in pc_df["origin_atoms"]]

    # Add pseudocenter IDs
    # Why? Some amino acids have several features of one features type, e.g. ARG has three HBD features.

    pc_ids = []

    # Initialize variables
    id_prefix_old = ""
    id_suffix = 1

    for index, row in pc_df.iterrows():

        # Create prefix of pseudocenter ID
        id_prefix_new = f'{row["residue"]}_{row["pc_type"]}'

        # Set suffix: starting with 1, but incrementing if prefix was seen before
        if id_prefix_new != id_prefix_old:
            id_suffix = 1
        else:
            id_suffix = id_suffix + 1

        # Add suffix to prefix
        pc_ids.append(f'{id_prefix_new}_{id_suffix}')

        # Update prefix
        id_prefix_old = id_prefix_new

    # Add pseudocenter IDs to DataFrame
    pc_df.insert(loc=0, column='pc_id', value=pc_ids)

    return pc_df


def _preprocess_pseudocenter_atoms():
    """
    Preprocess pseudocenter csv file as a DataFrame containing per row one pseudocenter atom with columns for
    - pseudocenter atom ID, e.g. ASN_HBA_1_OD1,
    - pseudocenter atom pattern (residue and atom name) e.g. ASN_OD1,
    - pseudocenter ID, e.g. ASN_HBA_1, and
    - pseudocenter type, e.g. HBA.

    Returns
    -------
    pandas.DataFrame
        Pseudocenter data (pseudocenter atom ID, pseudocenter atom pattern, pseudocenter ID, and pseudocenter type).
    """

    # Load pseudocenter DataFrame
    pc_df = _preprocess_pseudocenters()

    # Define a list of pseudocenter atoms
    pc_atom_ids = []  # Pseudocenter atom ID
    pc_atom_pattern = []  # Pseudocenter atom pattern
    pc_ids = []  # Pseudocenter ID
    pc_types = []  # Pseudocenter type

    for index, row in pc_df.iterrows():
        for j in row["origin_atoms"]:  # Some pseudocenters consist of several atoms
            pc_atom_pattern.append(f'{row["residue"]}_{j}')
            pc_types.append(row["pc_type"])
            pc_ids.append(row["pc_id"])
            pc_atom_ids.append(f'{row["pc_id"]}_{j}')

    # Save to dict
    pc_atoms = {
        "pc_atom_id": pc_atom_ids,
        "pc_atom_pattern": pc_atom_pattern,
        "pc_id": pc_ids,
        "pc_type": pc_types
    }

    # Cast dictionary to DataFrame
    pc_atoms_df = pd.DataFrame.from_dict(pc_atoms)

    # Save pseudocenter atoms to pickle file
    pc_atoms_df.to_csv(ratar_path / "data" / "pseudocenter_atoms.csv")

    return pc_atoms_df


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
