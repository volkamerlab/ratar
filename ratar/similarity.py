########################################################################################
# Binding site similarity
########################################################################################

# This script contains classes and functions used during the binding site similarity step.

########################################################################################
# Import modules
########################################################################################

import math
import numpy as np
import pandas as pd
from scipy.special import cbrt
from scipy.stats.stats import skew

import matplotlib.pyplot as plt
import seaborn as sns


########################################################################################
# Global variables
########################################################################################

# Project location
project_path: str = "/home/dominique/Documents/projects/readacross_targetome"


########################################################################################
# functions
########################################################################################

def get_similarity(moments_p1, moments_p2, measure):
    """
    Calculate the similarity between two proteins p1 and p2:
    - 1: inverse of the translated and scaled Manhattan distance
    - 2: tba
    """
    
    # calculate inverse of the translated and scaled Manhattan distance
    if measure == 1:
        return 1 / ( 1 + 1/moments_p1.size * abs(moments_p1-moments_p2).values.sum() )
    # print message if measure unknown
    else:
        print("Please choose a similarity measure.")


def get_similarity_matrix(moments, pdb_ids):
    """
    Calculate the similarity matrix (all pairwise comparisons between moments).
    Return as Pandas DataFrame.
    """
    
    # initialize the similarity matrix
    sim_matrix = []

    # treat each target as query (for loop)
    for query in moments:
        # calculate the similarity to each of the other targets (list comprehension)
        sim = [get_similarity(query, i, 1) for i in moments]
        # concatenate all query comparisons
        sim_matrix.append(sim)
    
    return pd.DataFrame(sim_matrix, columns=pdb_ids, index=pdb_ids)


def rank_targets_by_similarity(similarity_matrix, query_id):
    query = pd.DataFrame(similarity_matrix[query_id])
    targets_ranked = query.sort_values(by=query.columns[0], axis=0, ascending=False)
    return targets_ranked


########################################################################################
# functions
# to plot data
########################################################################################


def plot_similarity_matrix(sim_mat):
    """
    Plot similarity matrix.
    """
    
    fig = plt.figure(figsize=(10, 10))
    ax = sns.heatmap(sim_mat, cmap="Blues")
    ax.set_title("Binding site similarity matrix for sc-PDB EGFRs")

