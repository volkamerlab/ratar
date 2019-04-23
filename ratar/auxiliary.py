########################################################################################
# Helper functions
########################################################################################

# This script contains helper functions and classes.


########################################################################################
# Import modules
########################################################################################

from biopandas.mol2 import PandasMol2, split_multimol2
import pandas as pd
import os
import sys


########################################################################################
# Global variables
########################################################################################

# Project location
package_path: str = sys.path[0]


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

    def __init__(self, input_mol2_path, output_log_path=None):
        self.input_mol2_path = input_mol2_path
        self.output_log_path = output_log_path
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
