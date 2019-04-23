########################################################################################
# Binding site encoding
########################################################################################

# This script contains auxiliary classes for the binding site encoding step.


########################################################################################
# Import modules
########################################################################################

from biopandas.mol2 import PandasMol2
from biopandas.mol2 import split_multimol2

import pandas as pd

import glob
import os
import pickle

import matplotlib as plt
import seaborn as sns
import math


########################################################################################
# Global variables
########################################################################################

# Project location
project_path = "/home/dominique/Documents/projects/readacross_targetome"


########################################################################################
# Load mol2 file
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


########################################################################################
# Miscellaneous
########################################################################################

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


class AminoAcidDescriptors:
    """
    Amino acid descriptors, e.g. Z-scales.
    """

    def __init__(self):
        # Z-scales taken from: https://github.com/Superzchen/iFeature/blob/master/codes/ZSCALE.py
        self.zscales = pd.read_csv(project_path+"/data/zscales.csv", index_col="aa3")


########################################################################################
# Save and load binding sites
########################################################################################

def save_binding_site(binding_site, output_path):
    """
    This function saves an encoded binding site to a pickle file in an output directory.

    :param binding_site: Encoded binding site.
    :type binding_site: encoding.BindingSite

    :param output_path: Path to output directory (without / at the end).
    :type output_path: String

    :return: Pickle file is saved; no return value.
    :rtype: None
    """

    output_bs_path = output_path + "/binding_sites/" + binding_site.pdb_id + ".p"
    pickle.dump(binding_site, open(output_bs_path, "wb"))


def get_binding_site_path(pdb, output_path):
    """
    This functions returns a binding site pickle path based on a path wildcard constructed from
    - a four-letter PDB ID and
    - an output directory that contains binding site pickle file(s) in a "binding_sites" directory:

    Wildcard: output_path + "/binding_sites/" + pdb + "*.p"

    :param pdb: Four-letter PDB ID.
    :type: String

    :param output_path: Path to output directory containing binding site pickle file(s) in a "binding_sites" directory.
    :type: String

    :return: Path to binding site pickle file.
    :rtype: String
    """
    # Define wildcard for path to pickle file
    bs_wildcard = output_path + "/binding_sites/" + pdb + "*.p"

    # Retrieve all paths that match the wildcard
    bs_path = glob.glob(bs_wildcard)

    # If wildcard matches no file, retry.
    if len(bs_path) == 0:
        print("Error: Your input path matches no file. Please retry.")
        print("\nYour input wildcard was the following: ")
        print(bs_wildcard)
        return None

    # If wildcard matches multiple files, retry.
    elif len(bs_path) > 1:
        print("Error: Your input path matches multiple files. Please select one of the following as input string: ")
        for i in bs_path:
            print(i)
        print("\nYour input wildcard was the following: ")
        print(bs_wildcard)
        return None

    # If wildcard matches one file, load file.
    else:
        bs_path = bs_path[0]
        return bs_path


def load_binding_site(binding_site_path):
    """
    This function loads an encoded binding site from a pickle file.

    :param binding_site_path: Path to binding site pickle file.
    :type: String

    :return: Encoded binding site.
    :rtype: encoding.BindingSite
    """

    # Retrieve all paths that match the input path
    bs_path = glob.glob(binding_site_path)

    # If input path matches no file, retry.
    if len(bs_path) == 0:
        print("Error: Your input path matches no file. Please retry.")
        print("\nYour input path was the following: ")
        print(bs_path)
        return None

    # If input path matches multiple files, retry.
    elif len(bs_path) > 1:
        print("Error: Your input path matches multiple files. Please select one of the following as input string: ")
        for i in bs_path:
            print(i)
        print("\nYour input path was the following: ")
        print(bs_path)
        return None

    # If input path matches one file, load file.
    else:
        bs_path = bs_path[0]
        binding_site = pickle.load(open(bs_path, "rb"))
        print("The following file was loaded: ")
        print(bs_path)
        return binding_site


def save_cgo_files(binding_site, output_path):
    """
    Generate CGO files containing reference points.
    """

    output_cgo_path = output_path + "/cgo_files"
    pdb_id = binding_site.pdb_id

    ref_points_colors = sns.color_palette("hls", 7)

    for repres in binding_site.shapes.shapes_dict.keys():
        for method in binding_site.shapes.shapes_dict[repres].keys():
            if method != "na":

                ref_points = binding_site.shapes.shapes_dict[repres][method]["ref_points"]

                filename = output_cgo_path + "/" + pdb_id[:4] + "_" + repres.replace("_coord", "") + "_" + method + ".py"
                cgo_file = open(filename, 'w')

                cgo_file.write("from pymol import *\n")
                cgo_file.write("import os\n")
                cgo_file.write("from pymol.cgo import *\n")

                size = str(1)

                cgo_file.write("obj = [\n")

                counter_colors = 0

                for index, row in ref_points.iterrows():

                    # Set sphere color

                    ref_points_color = list(ref_points_colors[counter_colors])
                    counter_colors = counter_colors + 1

                    cgo_file.write("\tCOLOR, "
                                   + str(ref_points_color[0]) + ", "
                                   + str(ref_points_color[1]) + ", "
                                   + str(ref_points_color[2]) + ", \n")

                    # Set sphere coordinates
                    cgo_file.write("\tSPHERE, "
                                   + str(row["x"]) + ", "
                                   + str(row["y"]) + ", "
                                   + str(row["z"]) + ", "
                                   + size + ",")

                    cgo_file.write("\n")

                cgo_file.write("]\ncmd.load_cgo(obj, '" + filename.split(".")[0].split("/")[-1] + "')")

                cgo_file.close()

sns.palplot(sns.color_palette("hls", 7))


########################################################################################
# Perform analysis on encoding
########################################################################################

def plot_dist_distribution_two(d1, d2, n1, n2):
    """
    Plot distance distributions from two proteins.
    """

    fig = plt.figure(figsize=(16, 4))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    ax = fig.add_subplot(121)
    for j in d1.columns:
        sns.distplot(d1[j], label=j)
    ax.set_title(n1)
    ax.set_xlabel('distance [A]')
    ax.set_ylabel('frequency [%]')
    ax.legend()

    ax = fig.add_subplot(122)
    for j in d2.columns:
        sns.distplot(d2[j], label=j)
    ax.set_title(n2)
    ax.set_xlabel('distance [A]')
    ax.set_ylabel('frequency [%]')
    ax.legend()


def plot_dist_distribution_all(d, n):
    """
    Plot distance distributions for all proteins in list.
    """

    ncols = 3
    # print(ncols)
    nrows = math.ceil(len(d) / ncols)
    # print(nrows)

    figsize_x = 16
    figsize_y = nrows * 4

    fig = plt.figure(figsize=(figsize_x, figsize_y))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i in range(0, len(d)):
        ax = fig.add_subplot(nrows, ncols, i + 1)

        dist = d[i]
        for j in dist.columns:
            sns.distplot(dist[j], label=j)

        ax.set_title(n[i])
        ax.set_xlabel('distance [A]')
        ax.set_ylabel('frequency [%]')
        ax.legend()


########################################################################################
# Perform statistics on input data
########################################################################################

def get_residue_composition(binding_sites):
    """

    :param binding_sites: List of encoded binding sites.
    :type: List of BindingSite objects

    :return:
    """

    counts_list = []
    for bs in binding_sites:
        mol = bs.mol

        # Get all residue names (unique), e.g. ASP100
        subst_unq = pd.Series(mol["subst_name"].unique().copy())

        # Remove all digits from string in order to get amino acid/ion/etc. names
        subst_aa = subst_unq.apply(lambda x: ''.join([i for i in x if not i.isdigit()]))

        # A few residue names contained only digits (so now they are empty strings)
        # Get full string name back
        subst_aa[subst_aa == ""] = subst_unq[subst_aa == ""]

        # Count name frequency
        counts = subst_aa.value_counts()
        counts.rename(bs.pdb_id, inplace=True)
        counts_list.append(counts)

    counts_df = pd.concat(counts_list, axis=1, sort=False)
    counts_df.fillna(0, inplace=True)

    return counts_df.astype(int)
