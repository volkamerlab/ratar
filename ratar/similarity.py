"""
similarity.py

Read-across the targetome -
An integrated structure- and ligand-based workbench for computational target prediction and novel tool compound design

Handles the primary functions for comparing binding sites.
"""

from collections import defaultdict
import itertools
import glob
from pathlib import Path
import pickle

import pandas as pd
from scipy.spatial import distance


def calculate_similarity(fingerprint1, fingerprint2, measure="modified_manhattan"):
    """
    Calculate the similarity between two fingerprints based on a similarity measure.

    Parameters
    ----------
    fingerprint1 : 1D array-like or pandas.DataFrame
        Fingerprint for molecule.
    fingerprint2 : 1D array-like or pandas.DataFrame
        Fingerprint for molecule.
    measure : str
        Similarity measurement method:
         - modified_manhattan (inverse of the translated and scaled Manhattan distance)

    Returns
    -------
    float
        Similarity value.
    """

    measures = ["modified_manhattan"]

    # Convert DataFrame into 1D array
    if isinstance(fingerprint1, pd.DataFrame):
        fingerprint1 = fingerprint1.values.flatten()
    if isinstance(fingerprint2, pd.DataFrame):
        fingerprint2 = fingerprint2.values.flatten()

    if len(fingerprint1) != len(fingerprint2):
        raise ValueError(f"Input fingerprints must be of same length.")

    if measure == measures[0]:
        # Calculate inverse of the translated and scaled Manhattan distance
        return 1 / (1 + 1 / len(fingerprint1) * distance.cityblock(fingerprint1, fingerprint2))
    else:
        raise ValueError(f'Please choose a similarity measure: {", ".join(measures)}')


def get_similarity_all_against_all(encoded_molecules_path, measure="modified_manhattan"):
    """
    Get all encoded molecules from a given output directory and calculate all-against-all matrices for each ratar
    encoding method.

    Parameters
    ----------
    encoded_molecules_path : str
        Absolute path to encoded molecules files (can contain wildcards).

    Returns
    -------
    dict of pandas.DataFrame
        All-against-all similarity matrix (DataFrame) for each encoding method (dictionary).
    """

    # Get list of all encoded binding site files
    path_list = glob.glob(encoded_molecules_path)

    # Initialise dictionary of matrices (to be returned from function)
    sim_matrices = {}

    # Get binding site ids and initialise all-against-all DataFrame with similarities of one (identical binding site)
    pdb_ids = []

    for path in path_list:
        with open(path, "rb") as f:
            bindingsite = pickle.load(f)
        pdb_ids.append(bindingsite.molecule.code)

    sim_df = pd.DataFrame(float(1), index=pdb_ids, columns=pdb_ids)

    # Get example encoded binding site to retrieve binding site data architecture
    with open(path_list[0], "rb") as f:
        bindingsite = pickle.load(f)

    # Initialise each encoding method with all-against-all DataFrame with similarities of one
    for repres in bindingsite.shapes.data.keys():
        for dim in bindingsite.shapes.data[repres].keys():
            for method in bindingsite.shapes.data[repres][dim].keys():
                # Set name for encoding method (representatives and method) and
                # use as key for dictionary of all-against-all matrices
                desc = f"{repres}_{dim}_{method}"

                # Add initial all-against-all matrices in dictionary
                sim_matrices[desc] = sim_df

    # Load all possible binding site pairs (to construct an upper triangular matrix)
    for path1, path2 in itertools.combinations(path_list, r=2):
        # Load binding site pair
        with open(path1, "rb") as f:
            bindingsite1 = pickle.load(f)
        with open(path2, "rb") as f:
            bindingsite2 = pickle.load(f)

        for repres in bindingsite1.shapes.data.keys():
            for dim in bindingsite1.shapes.data[repres].keys():
                for method in bindingsite1.shapes.data[repres][dim].keys():
                    # Set name for encoding method (representatives and method) and
                    # use as key for dictionary of all-against-all matrices
                    desc = f"{repres}_{dim}_{method}"

                    # Get binding site ids
                    id1 = bindingsite1.molecule.code
                    id2 = bindingsite2.molecule.code

                    # Get similarity value
                    sim = calculate_similarity(
                        bindingsite1.shapes.data[repres][dim][method].moments,
                        bindingsite2.shapes.data[repres][dim][method].moments,
                        measure,
                    )

                    # Save similarity similarity to matrix
                    sim_matrices[desc].at[id1, id2] = round(sim, 5)  # upper matrix triangle
                    sim_matrices[desc].at[id2, id1] = round(sim, 5)  # lower matrix triangle

    return sim_matrices


def get_similarity_pairs(pairs, encoded_molecules_path, measure="modified_manhattan"):
    """
    Calculate the similarity between pairs of binding sites for different encoding methods and
    return a pairs ID x encoding method DataFrame containing the similarity values.

    Parameters
    ----------
    pairs : pandas.DataFrame
        DataFrame with structure ID for pairs of molecules (given in columns 'molecule1' and 'molecule2').
    encoded_molecules_path : str or pathlib.Path
        Full path to file with encoded molecules with %s (placeholder) for structure ID.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing a matrix of similarity values (pair ID x similarity measurement)

    Notes
    -----
    Example:
    Two pairs of molecules:
    (molecule11, molecule12), (molecule21, molecule22)

    sim_dict = {'encoding_method1': [0.5, 0.9], 'encoding_method2': [0.55, 0.94]}
    pair_list = ['molecule11_molecule12', 'molecule21_molecule22']

    sim_df =
                                encoding_method1    encoding_method2
        molecule11_molecule12     0.5                 0.55
        molecule21_molecule22     0.9                 0.94
    """

    pairwise_similarities = defaultdict(list)

    for _, pair in pairs.iterrows():
        path1 = Path(encoded_molecules_path.replace("%", pair[0]))
        path2 = Path(encoded_molecules_path.replace("%", pair[1]))

        with open(path1, "rb") as f:
            bindingsite1 = pickle.load(f)

        with open(path2, "rb") as f:
            bindingsite2 = pickle.load(f)

        shapes1 = bindingsite1.shapes.all
        shapes2 = bindingsite2.shapes.all

        if len(set(shapes1.keys()) - set(shapes2.keys())) > 0:
            raise ValueError(
                f"Input pair of encoded molecules does not contain the same encoding methods. "
                f"Please check."
            )

        for (shape_key1, shape1), (_, shape2) in zip(shapes1.items(), shapes2.items()):
            similarity = calculate_similarity(shape1.moments, shape2.moments, measure)
            pairwise_similarities[shape_key1].append(similarity)

    pairwise_similarities = pd.DataFrame(pairwise_similarities)
    pairwise_similarities.index = pd.MultiIndex.from_arrays(pairs.transpose().values)

    return pairwise_similarities
