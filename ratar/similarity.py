"""
similarity.py

Read-across the targetome -
An integrated structure- and ligand-based workbench for computational target prediction and novel tool compound design

Handles the primary functions for comparing binding sites.
"""


########################################################################################
# Import modules
########################################################################################

import sys
import pandas as pd

########################################################################################
# Global variables
########################################################################################

# Package location
# package_path: str = sys.path[0]
package_path: str = "/home/dominique/Documents/projects/ratar/ratar"
print(package_path)


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


def get_similarity_matrix(moments, pdb_ids):
    """
    Calculate the similarity matrix (all pairwise comparisons between moments).
    Return as Pandas DataFrame.

    :param moments:
    :param pdb_ids:
    :return:
    """

    # Initialize the similarity matrix
    sim_matrix = []

    # Treat each target as query (for loop)
    for query in moments:
        # Calculate the similarity to each of the other targets (list comprehension)
        sim = [get_similarity(query, i, 1) for i in moments]
        # Concatenate all query comparisons
        sim_matrix.append(sim)

    return pd.DataFrame(sim_matrix, columns=pdb_ids, index=pdb_ids)


def rank_targets_by_similarity(similarity_matrix, query_id):
    """

    :param similarity_matrix:
    :param query_id:
    :return:
    """
    query = pd.DataFrame(similarity_matrix[query_id])
    targets_ranked = query.sort_values(by=query.columns[0], axis=0, ascending=False)
    return targets_ranked
