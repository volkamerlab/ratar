"""
similarity.py

Read-across the targetome -
An integrated structure- and ligand-based workbench for computational target prediction and novel tool compound design

Handles the primary functions for comparing binding sites.
"""


########################################################################################
# Import modules
########################################################################################

from encoding import *

import glob
import pickle
import pandas as pd

########################################################################################
# Global variables
########################################################################################

# Package location
# package_path: str = sys.path[0]
package_path: str = "/home/dominique/Documents/projects/ratar/ratar"


########################################################################################
# Functions
########################################################################################

def get_similarity(moments_p1, moments_p2, measure):
    """
    Calculate the similarity between two proteins p1 and p2:
    - 1: inverse of the translated and scaled Manhattan distance
    - 2: tba

    :param moments_p1:
    :param moments_p2:
    :param measure:
    :return:
    """

    # Calculate inverse of the translated and scaled Manhattan distance
    if measure == 1:
        return 1 / (1 + 1 / moments_p1.size * abs(moments_p1 - moments_p2).values.sum())
    # Print message if measure unknown
    else:
        print("Please choose a similarity measure.")


def get_similarity_all_against_all(output_dir):
    """
    This function retrieves all encoded binding sites from a given output directory and
    calculates all-against-all matrices for each ratar encoding method.

    :param output_dir: Absolute path to directory containing the ratar encoding output.
    :type output_dir: String

    :return: All-against-all similarity matrix (DataFrame) for each encoding method (dictionary).
    :rtype: Dict of DataFrames
    """

    # Get list of all encoded binding site files
    file_list = glob.glob("%s/encoding/*/ratar_encoding.p" % output_dir)

    # Initialise dictionary of matrices (to be returned from function)
    sim_matrices = {}

    # Get binding site ids and initialise all-against-all DataFrame with similarities of one (identical binding site)
    pdb_ids = []
    for f in file_list:
        bs = pickle.load(open(f, "rb"))
        pdb_ids.append(bs.pdb_id)
    sim_df = pd.DataFrame(float(1), index=pdb_ids, columns=pdb_ids)

    # Get example encoded binding site to retrieve binding site data architecture
    bs = pickle.load(open(file_list[0], "rb"))

    # Initialise each encoding method with all-against-all DataFrame with similarities of one
    for repres in bs.shapes.shapes_dict.keys():
        for method in bs.shapes.shapes_dict[repres].keys():
            if method != "na":

                # Set name for encoding method (representatives and method) and
                # use as key for dictionary of all-against-all matrices
                desc = "%s_%s" % (repres, method)

                # Add initial all-against-all matrices in dictionary
                sim_matrices[desc] = sim_df

    # Load all possible binding site pairs (to construct an upper triangular matrix)
    for i in range(0, len(file_list) - 1):
        for j in range(i + 1, len(file_list)):

            # Load binding site pair
            bs1 = pickle.load(open(file_list[i], "rb"))
            bs2 = pickle.load(open(file_list[j], "rb"))

            for repres in bs1.shapes.shapes_dict.keys():
                for method in bs1.shapes.shapes_dict[repres].keys():
                    if method != "na":

                        # Set name for encoding method (representatives and method) and
                        # use as key for dictionary of all-against-all matrices
                        desc = "%s_%s" % (repres, method)

                        # Get binding site ids
                        id1 = bs1.pdb_id
                        id2 = bs2.pdb_id

                        # Get similarity value
                        sim = get_similarity(bs1.shapes.shapes_dict[repres][method]["moments"],
                                             bs2.shapes.shapes_dict[repres][method]["moments"],
                                             1)

                        # Save similarity similarity to matrix
                        sim_matrices[desc].at[id1, id2] = round(sim, 5)  # upper matrix triangle
                        sim_matrices[desc].at[id2, id1] = round(sim, 5)  # lower matrix triangle

    return sim_matrices


def rank_targets_by_similarity(similarity_matrix, query_id):
    """

    :param similarity_matrix:
    :param query_id:
    :return:
    """
    query = pd.DataFrame(similarity_matrix[query_id])
    targets_ranked = query.sort_values(by=query.columns[0], axis=0, ascending=False)
    return targets_ranked
