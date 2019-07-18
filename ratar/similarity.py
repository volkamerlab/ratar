"""
similarity.py

Read-across the targetome -
An integrated structure- and ligand-based workbench for computational target prediction and novel tool compound design

Handles the primary functions for comparing binding sites.

"""


import glob
import pickle

import pandas as pd


def get_similarity(moments_p1, moments_p2, measure):
    """
    Calculate the similarity between two proteins p1 and p2 based on a similarity measure.

    Parameters
    ----------
    moments_p1 : pandas.DataFrame
        Fingerprint for protein (binding site) p1.
    moments_p2 : pandas.DataFrame
        Fingerprint for protein (binding site) p2.
    measure : int
        Similarity measurement method:
         - 1 (inverse of the translated and scaled Manhattan distance)

    Returns
    -------
    float
        Similarity value.
    """

    # Calculate inverse of the translated and scaled Manhattan distance
    if measure == 1:
        return 1 / (1 + 1 / moments_p1.size * abs(moments_p1 - moments_p2).values.sum())
    # Print message if measure unknown
    else:
        print('Please choose a similarity measure: 1 (inverse of the translated and scaled Manhattan distance).')


def get_similarity_all_against_all(output_path):

    """
    Get all encoded binding sites from a given output directory and calculate all-against-all matrices for each ratar
    encoding method.

    Parameters
    ----------
    output_path : str
        Absolute path to encoded binding site files (can contain wildcards).

    Returns
    -------
    dict of pandas.DataFrame
        All-against-all similarity matrix (DataFrame) for each encoding method (dictionary).
    """

    # Get list of all encoded binding site files
    file_list = glob.glob(output_path)

    # Initialise dictionary of matrices (to be returned from function)
    sim_matrices = {}

    # Get binding site ids and initialise all-against-all DataFrame with similarities of one (identical binding site)
    pdb_ids = []

    for file in file_list:
        with open(file, 'rb') as f:
            bs = pickle.load(f)
        pdb_ids.append(bs.pdb_id)

    sim_df = pd.DataFrame(float(1), index=pdb_ids, columns=pdb_ids)

    # Get example encoded binding site to retrieve binding site data architecture
    with open(file_list[0], 'rb') as f:
        bs = pickle.load(f)

    # Initialise each encoding method with all-against-all DataFrame with similarities of one
    for repres in bs.shapes.shapes_dict.keys():
        for method in bs.shapes.shapes_dict[repres].keys():
            if method != 'na':

                # Set name for encoding method (representatives and method) and
                # use as key for dictionary of all-against-all matrices
                desc = f'{repres}_{method}'

                # Add initial all-against-all matrices in dictionary
                sim_matrices[desc] = sim_df

    # Load all possible binding site pairs (to construct an upper triangular matrix)
    for i in range(0, len(file_list) - 1):
        for j in range(i + 1, len(file_list)):

            # Load binding site pair
            with open(file_list[i], 'rb') as f:
                bs1 = pickle.load(f)
            with open(file_list[j], 'rb') as f:
                bs2 = pickle.load(f)

            for repres in bs1.shapes.shapes_dict.keys():
                for method in bs1.shapes.shapes_dict[repres].keys():
                    if method != 'na':

                        # Set name for encoding method (representatives and method) and
                        # use as key for dictionary of all-against-all matrices
                        desc = '{repres}_{method}'

                        # Get binding site ids
                        id1 = bs1.pdb_id
                        id2 = bs2.pdb_id

                        # Get similarity value
                        sim = get_similarity(bs1.shapes.shapes_dict[repres][method]['moments'],
                                             bs2.shapes.shapes_dict[repres][method]['moments'],
                                             1)

                        # Save similarity similarity to matrix
                        sim_matrices[desc].at[id1, id2] = round(sim, 5)  # upper matrix triangle
                        sim_matrices[desc].at[id2, id1] = round(sim, 5)  # lower matrix triangle

    return sim_matrices


def calculate_similarity_pairs(pairs, struc_path_template):

    """
    Calculate the similarity between pairs of binding sites for different encoding methods and
    return a pairs ID x encoding method DataFrame containing the similarity values.

    Parameters
    ----------
    pairs : pandas.DataFrame
        DataFrame with structure ID for pairs of binding sites (given in columns 'struc1' and 'struc2').
    struc_path_template : str
        Full path to file with encoded binding sites with %s (placeholder) for structure ID.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing a matrix of similarity values (pair ID x similarity measurement)

    Notes
    -----
    Example:
    Two pairs of binding sites: (p11, p12), (p21, p22)

    sim_dict = {'encoding_method1': [0.5, 0.9], 'encoding_method2': [0.55, 0.94]}
    pair_list = ['p11_p12', 'p21_p22']

    sim_df =
                    encoding_method1    encoding_method2
        p11_p12     0.5                 0.55
        p21_p22     0.9                 0.94
    """

    # Initialise objects to be filled while iterating over pairs
    sim_dict = {}  # This dictionary will be transformed to a DataFrame in the end
    pair_list = []  # This list will serve as index for that DataFrame

    # Initialise sim_dict with encoding method type (as keys) and empty lists (as values)

    # Load example binding site
    with open(struc_path_template % pairs.loc[0, 'struc1'], 'rb') as f:
        bs = pickle.load(f)

    for repres in bs.shapes.shapes_dict.keys():
        for method in bs.shapes.shapes_dict[repres].keys():
            if method != 'na':

                # Set name for encoding method (representatives and method) and
                # use as key for dictionary
                desc = f'{repres}_{method}'
                sim_dict[desc] = []

    for i in pairs.index:

        p1 = pairs.loc[i, 'struc1']  # Get one partner of pair
        p2 = pairs.loc[i, 'struc2']  # Get other partner of pair

        pair_list.append(f'{p1}_{p2}')  # Save pair as string

        # Get path to structure files
        struc_path1 = struc_path_template % p1
        struc_path2 = struc_path_template % p2

        # Load binding sites
        with open(struc_path1, 'rb') as f:
            bs1 = pickle.load(f)
        with open(struc_path2, 'rb') as f:
            bs2 = pickle.load(f)

        for repres in bs1.shapes.shapes_dict.keys():
            for method in bs1.shapes.shapes_dict[repres].keys():
                if method != 'na':

                    # Set name for encoding method (representatives and method) and
                    # use as key for dictionary
                    desc = f'{repres}_{method}'

                    # Get similarity value
                    sim = get_similarity(bs1.shapes.shapes_dict[repres][method]['moments'],
                                         bs2.shapes.shapes_dict[repres][method]['moments'],
                                         1)

                    # Add similarity value to similarity dictionary
                    sim_dict[desc].append(sim)

    # Transform dictionary of lists to DataFrame
    sim_df = pd.DataFrame(sim_dict, index=pair_list)

    return sim_df
