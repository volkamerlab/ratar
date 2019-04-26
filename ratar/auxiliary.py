"""
auxiliary.py

Read-across the targetome -
An integrated structure- and ligand-based workbench for computational target prediction and novel tool compound design

Handles the helper functions.
"""


########################################################################################
# Import modules
########################################################################################

from biopandas.mol2 import PandasMol2, split_multimol2
from biopandas.pdb import PandasPdb
import pandas as pd
import os


########################################################################################
# Global variables
########################################################################################

# Project location
# package_path: str = sys.path[0]
package_path: str = "/home/dominique/Documents/projects/ratar/ratar"


########################################################################################
# Load auxiliary data
########################################################################################

class AminoAcidDescriptors:
    """
    Amino acid descriptors, e.g. Z-scales.
    """

    def __init__(self):
        # Z-scales taken from: https://github.com/Superzchen/iFeature/blob/master/codes/ZSCALE.py
        self.zscales = pd.read_csv(package_path+"/data/zscales.csv", index_col="aa3")


########################################################################################
# IO functions
########################################################################################

class Mol2Loader:
    """
    Load binding sites from file(s).
    """

    def __init__(self, input_mol2_path):
        self.input_mol2_path = input_mol2_path
        self.pmols = self.read_from_mol2()

    def read_from_mol2(self):
        """
        Read the content of a mol2 file containing multiple entries.
        :param: Path to mol2 file.
        :return: List of BioPandas pmol objects.
        """

        pmols = []

        # In case of multiple entries in one mol2 file, include iteration step
        for mol2 in split_multimol2(self.input_mol2_path):
            pmol = PandasMol2().read_mol2_from_list(mol2_lines=mol2[1], mol2_code=mol2[0])
            pmols.append(pmol)

        return pmols


class MolFileLoader:
    """
    Load binding sites from file(s).
    """

    def __init__(self, input_path):
        self.input_path = input_path
        self.pmols = None

        file_suffix = os.path.basename(input_path).split(".")[-1]

        if file_suffix == "pdb":
            self.pmols = self.read_from_pdb()
        elif file_suffix == "mol2":
            self.pmols = self.read_from_mol2()

    def read_from_mol2(self):
        """
        Read the content of a mol2 file containing multiple entries.
        :param: Path to mol2 file.
        :return: List of BioPandas pmol objects.
        """

        # In case of multiple entries in one mol2 file, include iteration step
        pmols = []

        for mol2 in split_multimol2(self.input_path):
            pmol = PandasMol2().read_mol2_from_list(mol2_lines=mol2[1], mol2_code=mol2[0])

            pmol._df = pmol.df.loc[:, ["atom_id",
                                       "atom_name",
                                       "subst_name",
                                       "x",
                                       "y",
                                       "z",
                                       "charge"]]

            # Insert needed columns
            pmol.df.insert(loc=2, column="residue_id", value=[i[3:] for i in pmol.df["subst_name"]])
            pmol.df.insert(loc=3, column="residue_name", value=[i[:3] for i in pmol.df["subst_name"]])

            pmols.append(pmol)

        return pmols

    def read_from_pdb(self):

        pmol = PandasPdb().read_pdb(self.input_path)

        if pmol.code == "":
            pmol.code = os.path.basename(pdb_path).split(".")[0]

        # Get both ATOM and HETATM lines of PDB file
        pmol._df = pd.concat([pmol.df["ATOM"], pmol.df["HETATM"]])

        # Get needed columns
        pmol._df = pmol.df.loc[:,
                   ["atom_number", "atom_name", "residue_number", "residue_name", "x_coord", "y_coord", "z_coord",
                    "charge"]]

        # Insert needed columns
        pmol.df.insert(loc=4, column="subst_name",
                       value=["%s%s" % (i, j) for i, j in zip(pmol.df["residue_name"], pmol.df["residue_number"])])

        # Rename columns
        pmol.df.rename(index=str, inplace=True, columns={"atom_number": "atom_id",
                                                         "residue_number": "residue_id",
                                                         "x_coord": "x",
                                                         "y_coord": "y",
                                                         "z_coord": "z"})

        pmols = [pmol]  # This has no meaning for pdb and is only for mol2 files that have multiple entries

        return pmols


def create_folder(directory):
    """

    :param directory:
    :return:
    """

    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)
