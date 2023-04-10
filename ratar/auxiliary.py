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

import ratar

logger = logging.getLogger(__name__)


# Project location
ratar_path = Path(ratar.__file__).parent


class MoleculeLoader:
    """
    Class used to load molecule data from mol2 and pdb files in the form of unified BioPandas objects.

    Attributes
    ----------
    molecule_path : str or pathlib.Path
        Absolute path to a mol2 (can contain multiple entries) or pdb file.
    remove_solvent : bool
        Set True to remove solvent molecules (default: False).
    molecules : list of biopandas.mol2.pandas_mol2.PandasMol2 or biopandas.pdb.pandas_pdb.PandasPdb
        List of molecule data in the form of BioPandas objects.
    n_molecules : int
        Number of molecules loaded.

    Examples
    --------
    >>> from ratar.auxiliary import MoleculeLoader
    >>> molecule_path = '/path/to/pdb/or/mol2'
    >>> molecule_loader = MoleculeLoader()

    >>> molecules = molecule_loader.molecules  # Contains one or multiple molecule objects
    >>> molecule1 = molecules[0].df  # Molecule data
    >>> molecule1_id = molecules[0].code  # Molecule id

    >>> molecules[0].df == molecule_loader.molecules[0]
    True
    """

    def __init__(self, molecule_path, remove_solvent=False):
        self.molecule_path = Path(molecule_path)
        self.remove_solvent = remove_solvent
        self.molecules = self._load_molecule()
        self.n_molecules = len(self.molecules)

    def _load_molecule(self):
        """
        Load one or multiple molecules from pdb or mol2 file.

        Returns
        -------
        list of biopandas.mol2.pandas_mol2.PandasMol2 or biopandas.pdb.pandas_pdb.PandasPdb
            List of BioPandas objects containing metadata and structural data of molecule(s) in mol2 file.
        """

        if self.molecule_path.exists():
            logger.info(f"File to be loaded: {self.molecule_path}", extra={"molecule_id": "all"})
        else:
            logger.error(f"File not found: {self.molecule_path}", extra={"molecule_id": "all"})
            raise FileNotFoundError(f"File not found: {self.molecule_path}")

        # Load molecule data
        if self.molecule_path.suffix == ".pdb":
            molecules = self._load_pdb(self.remove_solvent)
        elif self.molecule_path.suffix == ".mol2":
            molecules = self._load_mol2(self.remove_solvent)
        else:
            raise IOError(
                f"Unsupported file format {self.molecule_path.suffix}, only pdb and mol2 are supported."
            )

        logger.info("File loaded.", extra={"molecule_id": "all"})

        return molecules

    @property
    def first_molecule(self):
        """
        Convenience class method: get the first molecule DataFrame.

        Returns
        -------
        biopandas.mol2.pandas_mol2.PandasMol2 or biopandas.pdb.pandas_pdb.PandasPdb
            Data for first molecule in MoleculeLoader class.
        """

        if len(self.molecules) > 0:
            return self.molecules[0]
        else:
            raise IndexError("MoleculeLoader.molecules is empty.")

    def _load_mol2(self, remove_solvent=False):
        """
        Load molecule data from a mol2 file, which can contain multiple entries.

        Parameters
        ----------
        remove_solvent : bool
            Set True to remove solvent molecules (default: False).

        Returns
        -------
        list of biopandas.mol2.pandas_mol2.PandasMol2
            List of BioPandas objects containing metadata and structural data of molecule(s) in mol2 file.
        """

        # In case of multiple entries in one mol2 file, include iteration step
        molecules = []

        for mol2 in split_multimol2(
            str(self.molecule_path)
        ):  # biopandas not compatible with pathlib
            # Mol2 files can have 9 or 10 columns.
            try:  # Try 9 columns.
                molecule = PandasMol2().read_mol2_from_list(
                    mol2_code=mol2[0],
                    mol2_lines=mol2[1],
                    columns={
                        0: ("atom_id", int),
                        1: ("atom_name", str),
                        2: ("x", float),
                        3: ("y", float),
                        4: ("z", float),
                        5: ("atom_type", str),
                        6: ("subst_id", str),
                        7: ("subst_name", str),
                        8: ("charge", float),
                    },
                )

            except ValueError:  # If 9 columns did not work, try 10 columns.
                molecule = PandasMol2().read_mol2_from_list(
                    mol2_code=mol2[0],
                    mol2_lines=mol2[1],
                    columns={
                        0: ("atom_id", int),
                        1: ("atom_name", str),
                        2: ("x", float),
                        3: ("y", float),
                        4: ("z", float),
                        5: ("atom_type", str),
                        6: ("subst_id", str),
                        7: ("subst_name", str),
                        8: ("charge", float),
                        9: ("status_bit", str),
                    },
                )

            # Insert additional columns (split ASN22 to ASN and 22)
            res_id_list = []
            res_name_list = []

            for i, j in zip(molecule.df["subst_name"], molecule.df["atom_type"]):
                if i[:2] == j.upper():
                    # These are ions such as CA or MG
                    res_id_list.append(i[2:])
                    res_name_list.append(i[:2])
                else:
                    # These are amino acid, linkers, compounds, ...
                    res_id_list.append(i[3:])
                    res_name_list.append(i[:3])

            molecule.df.insert(loc=2, column="res_id", value=res_id_list)
            molecule.df.insert(loc=2, column="res_name", value=res_name_list)

            # Select columns of interest
            molecule._df = molecule.df.loc[
                :,
                [
                    "atom_id",
                    "atom_name",
                    "res_id",
                    "res_name",
                    "subst_name",
                    "x",
                    "y",
                    "z",
                    "charge",
                ],
            ]

            # Remove solvent if parameter remove_solvent=True
            if remove_solvent:
                ix = molecule.df.index[molecule.df["res_name"] == "HOH"]
                molecule.df.drop(index=ix, inplace=True)

            molecules.append(molecule)

        return molecules

    def _load_pdb(self, remove_solvent=False):
        """
        Load molecule data from a pdb file, which can contain multiple entries.

        Parameters
        ----------
        remove_solvent : bool
            Set True to remove solvent molecules (default: False).

        Returns
        -------
        list of biopandas.pdb.pandas_pdb.PandasPdb
            List of BioPandas objects containing metadata and structural data of molecule(s) in pdb file.
        """

        molecule = PandasPdb().read_pdb(
            str(self.molecule_path)
        )  # biopandas not compatible with pathlib

        # If object has no code, set string from file stem and its folder name
        # E.g. "/mydir/pdb/3w32.mol2" will generate the code "pdb_3w32".
        if not (molecule.code or molecule.code.strip()):
            molecule.code = f"{self.molecule_path.parts[-2]}_{self.molecule_path.stem}"

        # Get both ATOM and HETATM lines of PDB file
        molecule._df = pd.concat([molecule.df["ATOM"], molecule.df["HETATM"]])

        # Select columns of interest
        molecule._df = molecule.df.loc[
            :,
            [
                "atom_number",
                "atom_name",
                "residue_number",
                "residue_name",
                "x_coord",
                "y_coord",
                "z_coord",
                "charge",
            ],
        ]

        # Insert additional columns
        molecule.df.insert(
            loc=4,
            column="subst_name",
            value=[
                f"{i}{j}"
                for i, j in zip(molecule.df["residue_name"], molecule.df["residue_number"])
            ],
        )

        # Rename columns
        molecule.df.rename(
            index=str,
            inplace=True,
            columns={
                "atom_number": "atom_id",
                "residue_number": "res_id",
                "residue_name": "res_name",
                "x_coord": "x",
                "y_coord": "y",
                "z_coord": "z",
            },
        )

        # Remove solvent if parameter remove_solvent=True
        if remove_solvent:
            ix = molecule.df.index[molecule.df["res_name"] == "HOH"]
            molecule.df.drop(index=ix, inplace=True)

        # Cast to list only for homogeneity with reading mol2 files that can have multiple entries
        molecules = [molecule]

        return molecules


class AminoAcidDescriptors:
    """
    Class used to store amino acid descriptor data, e.g. Z-scales.

    Attributes
    ----------
    zscales : pandas.DataFrame
        Z-scales for standard and a few non-standard amino acids.

    Examples
    --------
    >>> from ratar.auxiliary import MoleculeLoader, AminoAcidDescriptors

    >>> amino_acid_descriptors = AminoAcidDescriptors()

    >>> molecule_path = '/path/to/pdb/or/mol2'
    >>> molecule_loader = MoleculeLoader()
    >>> molecule_loader.load_molecule(molecule_path, remove_solvent=True)
    >>> molecule1 == molecule_loader.molecules[0]

    >>> molecule1_zscales = amino_acid_descriptors.get_zscales_amino_acids(molecule1)

    Notes
    -----
    Z-scales taken from: https://github.com/Superzchen/iFeature/blob/master/codes/ZSCALE.py
    """

    def __init__(self):
        zscales_path = ratar_path / "data" / "zscales.csv"
        self.zscales = pd.read_csv(zscales_path, index_col="aa3")

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
        mol_aa = molecule["res_name"]

        # Get only rows (atoms) that belong to Z-scales amino acids
        mol_zscales_aa = molecule[mol_aa.apply(lambda y: y in self.zscales.index)].copy()

        # Get only rows (atoms) that DO NOT belong to Z-scales amino acids
        mol_non_zscales_aa = molecule[mol_aa.apply(lambda y: y not in self.zscales.index)].copy()

        if not mol_non_zscales_aa.empty:
            logger.info(
                f"Atoms removed for binding site encoding: {mol_non_zscales_aa.to_string()}"
            )

        return mol_zscales_aa


def load_pseudocenters(remove_hbda=False):
    """
    Load pseudocenters from file.
    Remove HBDA features, since they contain too few data points for encoding.

    Parameters
    ----------
    remove_hbda : bool
        Set True if pseudocenter type HBDA shall be removed (default: False).

    Returns
    -------
    pandas.DataFrame
        DataFrame containing pseudocenter information.

    Examples
    --------
    >>> from ratar.auxiliary import load_pseudocenters
    >>> pseudocenter_atoms = load_pseudocenters()
    """

    # TODO summarize pseudocenter functions in Pseudocenter class?

    pseudocenter_path = ratar_path / "data" / "pseudocenter_atoms.csv"

    pseudocenter_atoms = pd.read_csv(pseudocenter_path, index_col=0)

    # Remove HBDA features information (too few data points)
    if remove_hbda:
        pseudocenter_atoms = pseudocenter_atoms[pseudocenter_atoms["pc_type"] != "HBDA"]
        pseudocenter_atoms.reset_index(drop=True, inplace=True)

    return pseudocenter_atoms


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
        pc_ids.append(f"{id_prefix_new}_{id_suffix}")

        # Update prefix
        id_prefix_old = id_prefix_new

    # Add pseudocenter IDs to DataFrame
    pc_df.insert(loc=0, column="pc_id", value=pc_ids)

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
        "pc_type": pc_types,
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
        raise OSError(f"Creating directory failed: {directory}")
