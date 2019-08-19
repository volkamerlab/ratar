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

from flatten_dict import flatten, unflatten
import pandas as pd
from scipy.spatial import distance

from ratar.auxiliary import MoleculeLoader


def calculate_similarity(fingerprint1, fingerprint2, measure):
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

    measures = ['modified_manhattan']

    # Convert DataFrame into 1D array
    if isinstance(fingerprint1, pd.DataFrame):
        fingerprint1 = fingerprint1.values.flatten()
    if isinstance(fingerprint2, pd.DataFrame):
        fingerprint2 = fingerprint2.values.flatten()

    if len(fingerprint1) != len(fingerprint2):
        raise ValueError(f'Input fingerprints must be of same length.')

    if measure == measures[0]:
        # Calculate inverse of the translated and scaled Manhattan distance
        return 1 / (1 + 1 / len(fingerprint1) * distance.cityblock(fingerprint1, fingerprint2))
    else:
        raise ValueError(f'Please choose a similarity measure: {", ".join(measures)}')


def get_similarity_all_against_all(encoded_molecules_path):

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
        with open(path, 'rb') as f:
            bindingsite = pickle.load(f)
        pdb_ids.append(bindingsite.pdb_id)

    sim_df = pd.DataFrame(float(1), index=pdb_ids, columns=pdb_ids)

    # Get example encoded binding site to retrieve binding site data architecture
    with open(path_list[0], 'rb') as f:
        bindingsite = pickle.load(f)

    # Initialise each encoding method with all-against-all DataFrame with similarities of one
    for repres in bindingsite.shapes.shapes_dict.keys():
        for method in bindingsite.shapes.shapes_dict[repres].keys():
            if method != 'na':

                # Set name for encoding method (representatives and method) and
                # use as key for dictionary of all-against-all matrices
                desc = f'{repres}_{method}'

                # Add initial all-against-all matrices in dictionary
                sim_matrices[desc] = sim_df

    # Load all possible binding site pairs (to construct an upper triangular matrix)
    for path1, path2 in itertools.combinations(path_list):

            # Load binding site pair
            with open(path1, 'rb') as f:
                bindingsite1 = pickle.load(f)
            with open(path2, 'rb') as f:
                bindingsite2 = pickle.load(f)

            for repres in bindingsite1.shapes.shapes_dict.keys():
                for method in bindingsite2.shapes.shapes_dict[repres].keys():
                    if method != 'na':

                        # Set name for encoding method (representatives and method) and
                        # use as key for dictionary of all-against-all matrices
                        desc = '{repres}_{method}'

                        # Get binding site ids
                        id1 = bindingsite1.pdb_id
                        id2 = bindingsite2.pdb_id

                        # Get similarity value
                        sim = calculate_similarity(bindingsite1.shapes.shapes_dict[repres][method]['moments'],
                                                   bindingsite2.shapes.shapes_dict[repres][method]['moments'],
                                                   1)

                        # Save similarity similarity to matrix
                        sim_matrices[desc].at[id1, id2] = round(sim, 5)  # upper matrix triangle
                        sim_matrices[desc].at[id2, id1] = round(sim, 5)  # lower matrix triangle

    return sim_matrices


def calculate_similarity_pairs(pairs, encoded_molecules_path):
    """
    Calculate the similarity between pairs of binding sites for different encoding methods and
    return a pairs ID x encoding method DataFrame containing the similarity values.

    Parameters
    ----------
    pairs : pandas.DataFrame
        DataFrame with structure ID for pairs of binding sites (given in columns 'molecule1' and 'molecule2').
    encoded_molecules_path : str or pathlib.Path
        Full path to file with encoded binding sites with %s (placeholder) for structure ID.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing a matrix of similarity values (pair ID x similarity measurement)

    Notes
    -----
    Example:
    Two pairs of binding sites: 
    (molecule11, molecule12), (molecule21, molecule22)

    sim_dict = {'encoding_method1': [0.5, 0.9], 'encoding_method2': [0.55, 0.94]}
    pair_list = ['molecule11_molecule12', 'molecule21_molecule22']

    sim_df =
                                encoding_method1    encoding_method2
        molecule11_molecule12     0.5                 0.55
        molecule21_molecule22     0.9                 0.94
    """

    # Initialise objects to be filled while iterating over pairs
    sim_dict = {}  # This dictionary will be transformed to a DataFrame in the end
    pair_list = []  # This list will serve as index for that DataFrame

    # Initialise sim_dict with encoding method type (as keys) and empty lists (as values)

    # Load example binding site
    with open(Path(encoded_molecules_path / pairs.loc[0, 'molecule1']), 'rb') as f:
        bindingsite = pickle.load(f)

    for repres in bindingsite.shapes.shapes_dict.keys():
        for method in bindingsite.shapes.shapes_dict[repres].keys():
            if method != 'na':

                # Set name for encoding method (representatives and method) and
                # use as key for dictionary
                desc = f'{repres}_{method}'
                sim_dict[desc] = []

    for i in pairs.index:

        p1 = pairs.loc[i, 'molecule1']  # Get one partner of pair
        p2 = pairs.loc[i, 'molecule2']  # Get other partner of pair

        pair_list.append(f'{p1}_{p2}')  # Save pair as string

        # Get path to structure files
        struc_path1 = encoded_molecules_path % p1
        struc_path2 = encoded_molecules_path % p2

        # Load binding sites
        with open(struc_path1, 'rb') as f:
            bindingsite1 = pickle.load(f)
        with open(struc_path2, 'rb') as f:
            bindingsite2 = pickle.load(f)

        for repres in bindingsite1.shapes.shapes_dict.keys():
            for method in bindingsite1.shapes.shapes_dict[repres].keys():
                if method != 'na':

                    # Set name for encoding method (representatives and method) and
                    # use as key for dictionary
                    desc = f'{repres}_{method}'

                    # Get similarity value
                    sim = calculate_similarity(bindingsite1.shapes.shapes_dict[repres][method]['moments'],
                                               bindingsite2.shapes.shapes_dict[repres][method]['moments'],
                                               1)

                    # Add similarity value to similarity dictionary
                    sim_dict[desc].append(sim)

    # Transform dictionary of lists to DataFrame
    sim_df = pd.DataFrame(sim_dict, index=pair_list)

    return sim_df


def get_similarity_pairs(pairs, encoded_molecules_path, measure):
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
    
    for index, pair in pairs.iterrows():

        path1 = Path(encoded_molecules_path.replace('%', pair[0]))
        path2 = Path(encoded_molecules_path.replace('%', pair[1]))

        with open(path1, 'rb') as f:
            bindingsite1 = pickle.load(f)

        with open(path2, 'rb') as f:
            bindingsite2 = pickle.load(f)

        shapes1 = bindingsite1.shapes.all
        shapes2 = bindingsite2.shapes.all

        if not set(shapes1.keys()) - set(shapes2.keys()):
            raise ValueError(f'Input pair of encoded molecules does not contain the same encoding methods. '
                             f'Please check.')

        pairwise_similarity_by_methods = []

        for shape_key, shape1 in shapes1.items():

            shape2 = shapes2[shape_key]

            pairwise_similarity_by_method = calculate_similarity(shape1.moments, shape2.moments, measure)
            pairwise_similarity_by_methods.append(pairwise_similarity_by_method)

        pairwise_similarities[shape_key] = pairwise_similarity_by_methods
    
    return pairwise_similarities


def load_flat_encoded_molecule(encoded_molecule_path, molecule_name):
    """

    Parameters
    ----------
    encoded_molecule_path : str or pathlib.Path
        Path to encoded molecule file.

    Returns
    -------
    pandas.DataFrame
        All encodings for the molecule: index = encodings name and column = molecule name.
    """

    with open(encoded_molecule_path, 'rb') as f:
        bindingsite = pickle.load(f)

    # Convert to DataFrame
    flat_encoded_molecule_df = pd.DataFrame.from_dict(flat_encoded_molecule, orient='index', columns=[molecule_name])

    return flat_encoded_molecule_df