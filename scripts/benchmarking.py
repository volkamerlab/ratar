"""
benchmarking.py

Read-across the targetome -
An integrated structure- and ligand-based workbench for computational target prediction and novel tool compound design

Handles the primary functions for benchmarking the ratar methodology.
"""

import pandas as pd
import pickle

from ratar.similarity import calculate_similarity_pairs


def get_similarity_pairs(benchmarkset):

    """
    Calculate the similarity values for binding site pairs described in different benchmarking datasets.

    Parameters
    ----------
    benchmarkset : str
        Benchmarking dataset type.

    Returns
    -------
    dict of pandas.DataFrame
        Dictionary of DataFrames containing a matrix of similarity values (pair ID x similarity measurement)

    Notes
    -----
    For a given benchmark dataset, return a dictionary of DataFrames that contains each different similarity measures
    for pairs of binding sites.
    """

    benchmarksets = ["fuzcav", "tough-m1"]

    if benchmarkset == benchmarksets[0]:

        # Set path to pairs list
        sim_pairs_path = (
            "/home/dominique/Documents/data/benchmarking/fuzcav/sim_dis_pairs/similar_pairs.txt"
        )
        dis_pairs_path = (
            "/home/dominique/Documents/data/benchmarking/fuzcav/sim_dis_pairs/dissimilar_pairs.txt"
        )

        # Get pairs list
        sim_pairs = pd.read_csv(
            sim_pairs_path, delimiter="  ", names=["struc1", "struc2"], engine="python"
        )
        dis_pairs = pd.read_csv(
            dis_pairs_path, delimiter="  ", names=["struc1", "struc2"], engine="python"
        )

        # Set path to structures directory
        encoded_benchmarkset_path = (
            "/home/dominique/Documents/projects/ratar-data/results/benchmarking/"
            "fuzcav/sim_dis_pairs/encoding/%s_site/ratar_encoding.p"
        )

        similarity = {
            "sim_pairs": calculate_similarity_pairs(sim_pairs, encoded_benchmarkset_path),
            "dis_pairs": calculate_similarity_pairs(dis_pairs, encoded_benchmarkset_path),
        }

        # Save to file
        similarity_path = (
            "/home/dominique/Documents/projects/ratar-data/results/benchmarking/"
            "fuzcav/sim_dis_pairs/similarity/pairs_similarity.p"
        )
        with open(similarity_path, "wb") as f:
            pickle.dump(similarity, f)

        return similarity

    elif benchmarkset == benchmarksets[1]:

        # Set path to dataset
        sim_pairs_path = (
            "/home/dominique/Documents/data/benchmarking/TOUGH-M1/TOUGH-M1_positive.list"
        )
        dis_pairs_path = (
            "/home/dominique/Documents/data/benchmarking/TOUGH-M1/TOUGH-M1_negative.list"
        )

        # Get pairs
        sim_pairs = pd.read_csv(
            sim_pairs_path,
            delimiter=" ",
            names=["struc1", "struc2", "seq_sim", "struc_sim", "lig_sim"],
        )
        dis_pairs = pd.read_csv(
            dis_pairs_path,
            delimiter=" ",
            names=["struc1", "struc2", "seq_sim", "struc_sim", "lig_sim"],
        )

        # Set path to structures directory
        encoded_benchmarkset_path = (
            "/home/dominique/Documents/projects/ratar-data/results/benchmarking/"
            "TOUGH-M1/encoding/%s/ratar_encoding.p"
        )

        similarity = {
            "sim_pairs": calculate_similarity_pairs(sim_pairs, encoded_benchmarkset_path),
            "dis_pairs": calculate_similarity_pairs(dis_pairs, encoded_benchmarkset_path),
        }

        # Save to file
        similarity_path = (
            "/home/dominique/Documents/projects/ratar-data/results/benchmarking/"
            "TOUGH-M1/similarity/pairs_similarity.p"
        )
        with open(similarity_path, "wb") as f:
            pickle.dump(similarity, f)

        return similarity

    else:
        raise ValueError(
            f'Selected benchmarking dataset unknown. Please choose from: {", ".join(benchmarksets)}'
        )


def main():
    """
    Main function to calculate similarity pairs for different benchmarking datasets.
    """

    get_similarity_pairs("fuzcav")
    get_similarity_pairs("tough-m1")


if __name__ == "__main__":
    main()
