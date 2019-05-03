"""
encoding.py

Read-across the targetome -
An integrated structure- and ligand-based workbench for computational target prediction and novel tool compound design

Handles the primary functions for encoding a single binding site.
"""


from auxiliary import *

import _pickle as pickle
from typing import List

import re
import sys
import glob

import numpy as np
import pandas as pd
from scipy.special import cbrt
from scipy.stats.stats import skew

import seaborn as sns


########################################################################################
# Global variables
########################################################################################

# Package location
# package_path: str = sys.path[0]
package_path: str = "/home/dominique/Documents/projects/ratar/ratar"

# Representative and physicochemical property keys
repres_keys: List[str] = ["ca", "pca", "pc"]
pcprop_keys: List[str] = ["z1", "z12", "z123"]

# Pseudocenters definition
pc_atoms: pd.DataFrame = pickle.load(open(package_path+"/data/pseudocenter_atoms.p", "rb"))
pc_atoms = pc_atoms[pc_atoms["type"] != "HBDA"]  # remove HBDA features information (too few data points)
pc_atoms.reset_index(drop=True, inplace=True)

# Amino acid descriptors definition, e.g. Z-scales
aa = AminoAcidDescriptors()


########################################################################################
# Process binding sites
########################################################################################

def encode_binding_site(pmol, output_log_path=None):

    """
    Encode the binding site stored in a pmol object, and optionally save progress to a log file.

    Parameters
    ----------
    pmol : biopandas.mol2.pandas_mol2.PandasMol2
        Coordinates and PDB ID for one binding site.
    output_log_path: string
        Path to log file.

    Returns
    -------
    encoding.BindingSite
        Encoded binding site.
    """

    if output_log_path is not None:
        log_file = open(output_log_path, "w")
        log_file.write("%s\n\n" % pmol.code)
        log_file.write("Encode binding site.\n\n")
        log_file.close()

    # Encode binding site
    binding_site = BindingSite(pmol, output_log_path)

    return binding_site


########################################################################################
# Save encoding related files
########################################################################################

def save_binding_site(binding_site, output_path):

    """
    Save an encoded binding site to a pickle file in an output directory.

    Parameters
    ----------
    binding_site : encoding.BindingSite
        Encoded binding site.
    output_path : string
        Path to output file.

    Returns
    -------
    None

    Notes
    -----
    Pickle file is saved; no return value.

    """

    pickle.dump(binding_site, open(output_path, "wb"))


def save_cgo_file(binding_site, output_path):

    """
    Generate a CGO file containing reference points for different encoding methods.

    Parameters
    ----------

    binding_site : encoding.BindingSite
        Encoded binding site.
    output_path : string
        Path to output file.

    Returns
    -------
    None

    Notes
    -----
    Python script (cgo file) for PyMol.

    """

    # Set PyMol sphere colors (for reference points)
    sphere_colors = sns.color_palette("hls", 7)

    # Open cgo file
    cgo_file = open(output_path, 'w')
    cgo_file.write("from pymol import *\n")
    cgo_file.write("import os\n")
    cgo_file.write("from pymol.cgo import *\n\n")

    # Collect all PyMol objects here (in order to group them after loading them to PyMol)
    obj_names = []

    for repres in binding_site.shapes.shapes_dict.keys():
        for method in binding_site.shapes.shapes_dict[repres].keys():
            if method != "na":

                # Get reference points (coordinates)
                ref_points = binding_site.shapes.shapes_dict[repres][method]["ref_points"]

                # Set descriptive name for reference points (PDB ID, representatives, dimensions, encoding method)
                obj_name = "%s_%s_%s" % (binding_site.pdb_id[:4], repres, method)
                obj_names.append(obj_name)

                # Set size for PyMol spheres
                size = str(1)

                cgo_file.write("obj_%s = [\n" % obj_name)  # Variable cannot start with digit, thus add prefix "obj_"

                # Set color counter (since we iterate over colors for each reference point)
                counter_colors = 0

                # For each reference point, write sphere color, coordinates and size to file
                for index, row in ref_points.iterrows():

                    # Set sphere color
                    sphere_color = list(sphere_colors[counter_colors])
                    counter_colors = counter_colors + 1

                    # Write sphere color to file
                    cgo_file.write("\tCOLOR, "
                                   + str(sphere_color[0]) + ", "
                                   + str(sphere_color[1]) + ", "
                                   + str(sphere_color[2]) + ", \n")

                    # Write sphere coordinates and size to file
                    cgo_file.write("\tSPHERE, "
                                   + str(row["x"]) + ", "
                                   + str(row["y"]) + ", "
                                   + str(row["z"]) + ", "
                                   + size + ", \n")

                # Write command to file that will load the reference points as PyMol object
                cgo_file.write("]\ncmd.load_cgo(obj_%s, '%s')\n\n" % (obj_name, obj_name))

    # Group all objects to one group
    cgo_file.write("cmd.group('%s_ref_points', '%s')" % (binding_site.pdb_id[:4], " ".join(obj_names)))

    # Close cgo file
    cgo_file.close()


########################################################################################
# Load encoding related files
########################################################################################

def get_encoded_binding_site_path(pdb, output_path):

    """
    Get a binding site pickle path based on a path wildcard constructed from
    - a four-letter PDB ID and
    - an output directory that contains ratar related files:

    Wildcard: output_path + "/encoding/" + pdb + "*.p"

    Parameters
    ----------
    pdb: string
        Four-letter PDB ID.
    output_path: string
        Path to output directory containing ratar related files.

    Returns
    -------
    String
        Path to binding site pickle file.

    """
    # Define wildcard for path to pickle file
    bs_wildcard = "%s/encoding/%s/ratar_encoding.p" % (output_path, pdb)

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

    # If wildcard matches one file, return file path.
    else:
        bs_path = bs_path[0]
        return bs_path


def load_binding_site(binding_site_path):

    """
    Load an encoded binding site from a pickle file.

    Parameters
    ----------
    binding_site_path : string
        Path to binding site pickle file.

    Returns
    -------
    encoding.BindingSite
        Encoded binding site.

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


########################################################################################
# Classes
########################################################################################


class BindingSite:

    """
    Class used to represent a binding site, containing:

    - pdb_id: the PDB ID (or structure ID)
    - mol: DataFrame containing atom lines from input file
    - repres: Representatives instance
    - subset: Subsetter instance
    - coord: Coordinates instance
    - pcprop: PCProperties instance
    - points: Points instance
    - shapes: Shapes instance

    Attributes
    ----------
    pmol : biopandas.mol2.pandas_mol2.PandasMol2 or biopandas.pdb.pandas_pdb.PandasPdb
        Content of mol2 or pdb file as BioPandas object.
    output_log_path : str
        Path to output log file.

    Methods
    -------
    None
    """

    def __init__(self, pmol, output_log_path=None):

        self.pdb_id = pmol.code
        self.mol = get_zscales_amino_acids(pmol.df, output_log_path)
        self.repres = Representatives(self.mol)
        self.subset = Subsetter(self.repres.repres_dict)
        self.coord = Coordinates(self.repres.repres_dict)
        self.pcprop = PCProperties(self.repres.repres_dict, output_log_path)
        self.points = Points(self.coord.coord_dict, self.pcprop.pcprop_dict, self.subset.subsets_indices_dict)
        self.shapes = Shapes(self.points)


class Representatives:

    """
    Class used to store binding site representatives.
    Representatives are selected atoms in a binding site,
    for instances all Calpha atoms of a binding site could serve as its representatives.

    This class contains:
    - repres_dict: Representatives stored as dictionary with several representation methods serving as key.
      Example: {'ca': ..., 'pca': ..., 'pc': ...}

    Attributes
    ----------
    mol : pandas DataFrame
        DataFrame containing atom lines of mol2 or pdb file.

    Methods
    -------
    None
    """

    def __init__(self, mol):

        self.repres_dict = {}
        for repres_key in repres_keys:
            self.repres_dict[repres_key] = get_representatives(mol, repres_key)


class Coordinates:

    """
    Class used to store the coordinates of the binding site representatives,
    which were defined by the Representatives class (in its repres_dict variable).

    This class contains:
    - coord_dict: Coordinates stored as dictionary with the same keys as in Representatives.repres_dict.
      Example: {'ca': ..., 'pca': ..., 'pc': ...}

    Attributes
    ----------
    repres_dict : Representatives.repres_dict (dictionary)
        Dictionary with several representation methods serving as key.

    Methods
    -------
    None
    """

    def __init__(self, repres_dict):

        self.coord_dict = {}
        for i in repres_keys:
            if type(repres_dict[i]) is not dict:
                self.coord_dict[i] = repres_dict[i][["x", "y", "z"]]
            else:
                self.coord_dict[i] = {key: value[["x", "y", "z"]] for (key, value) in repres_dict[i].items()}


class PCProperties:

    """
    Class used to store the physicochemical properties of binding site representatives,
    which were defined by the Representatives class (in its repres_dict variable).

    This class contains:
    - pcprop_dict: Physicochemical properties stored as dictionary with the same keys as in Representatives.repres_dict.
      Example: {'ca': ..., 'pca': ..., 'pc': ...}
    - output_log_path: Path to output log file.

    Coordinates are stored as dictionary (coord_dict), with the same keys as in Representatives.repres_dict.

    Attributes
    ----------
    repres_dict : Representatives.repres_dict (dictionary)
        Dictionary with several representation methods serving as key.
    output_log_path : str
        Path to output log file.

    Methods
    -------
    None
    """

    def __init__(self, repres_dict, output_log_path=None):

        self.pcprop_dict = {}
        self.output_log_path = output_log_path

        for i in repres_keys:
            self.pcprop_dict[i] = {}
            for j in pcprop_keys:
                if type(repres_dict[i]) is not dict:
                    self.pcprop_dict[i][j] = get_pcproperties(repres_dict, i, j)
                else:
                    self.pcprop_dict[i] = {key: get_pcproperties(repres_dict, i, j)
                                           for (key, value) in repres_dict[i].items()}


class Subsetter:

    """
    Class used to store subsets of binding site representatives,
    which were defined by the Representatives class (in its repres_dict variable).

    This class contains:
    - subsets_indices_dict: Subsets stored as dictionary with the same keys as in Representatives.repres_dict.
      Example: {'ca': {'H': ..., 'HBD': ..., ...},
                'pca': {'H': ..., 'HBD': ..., ...},
                'pc': {'H': ..., 'HBD': ..., ...}}

    Attributes
    ----------
    repres_dict : Representatives.repres_dict (dictionary)
        Dictionary with several representation methods serving as key.

    Methods
    -------
    None
    """

    def __init__(self, repres_dict):

        self.subsets_indices_dict = {"pc": get_subset_indices(repres_dict, "pc"),
                                     "pca": get_subset_indices(repres_dict, "pca")}


class Points:

    """
    Class used to store the vectors for the binding site representatives,
    which were defined by the Representatives class (in its repres_dict variable).

    Binding site representatives (i.e. atoms) can have different dimensionalities, for instance
    an atom can have
    - 3 dimensions (spatial properties x, y, z) or
    - more dimensions (spatial and some additional properties).

    This class contains:
    - points_dict: 3- to n-dimensional vectors for binding site representatives
      Example: {'ca': ..., 'ca_z1': ..., 'ca_z123': ..., ..., 'pca': ..., ...}
    - points_subsets_dict: 3- to n-dimensional vectors for binding site representatives, grouped by subsets.
      Example: {'pc_z1': {'H': ..., 'HBD': ..., ...},
                'pc_z12': {'H': ..., 'HBD': ..., ...},
                ...,
                'pca_z12': {'H': ..., 'HBD': ..., ...},
                ...}

    Attributes
    ----------
    coord_dict: Coordinates.coord_dict (dictionary)
        Dictionary with spatial properties (=coordinates) for each representative.
        Has the same keys as Representatives.repres_dict.
    pcprop_dict: PCProperties.pcprop_dict (dictionary)
        Dictionary with physicochemical properties for each representative.
        Has the same top level keys as Representatives.repres_dict,
        with nested keys describing different physicochemical properties per representative type.
    subsets_indices_dict: Subsetter.subsets_indices_dict (dictionary)
        Dictionary with subsets of representatives.
        Has the same top level keys as Representatives.repres_dict,
        with nested keys describing different subsetting types.

    Methods
    -------
    None
    """

    def __init__(self, coord_dict, pcprop_dict, subsets_indices_dict):

        self.points_dict = get_points(coord_dict, pcprop_dict)
        self.points_subsets_dict = get_points_subsetted(self.points_dict, subsets_indices_dict)


class Shapes:

    """
    Class used to store the encoded binding site representatives,
    which were defined by the Representatives class (in its repres_dict variable).

    This class contains:
    - shapes_dict: Encoding stored as dictionary with
        - level 1 keys for representatives, e.g. 'ca',
        - level 2 keys for encoding method, e.g. '3dim_usr',
        - level 3 keys for reference point coordinates 'ref_points', distances 'dist', and moments 'moments'.
    - shapes_subsets_dict: Encoding stored as dictionary
        - level 1 keys for representatives, e.g. 'ca',
        - level 2 keys for subsets, e.g. 'H',
        - level 3 keys for encoding method, e.g. '3dim_usr',
        - level 4 keys for reference point coordinates 'ref_points', distances 'dist', and moments 'moments'.

    Attributes
    ----------
    points: Points.points (dictionary)
        Dictionary with spatial properties (=coordinates) for each representative.
        Has the same keys as Representatives.repres_dict.

    Methods
    -------
    None
    """

    def __init__(self, points):

        # Full datasets
        self.shapes_dict = {}
        for key in points.points_dict:
            self.shapes_dict[key] = get_shape(points.points_dict[key])

        # Subset datasets
        self.shapes_subsets_dict = {}
        for ko in points.points_subsets_dict:  # ko is outer key
            self.shapes_subsets_dict[ko] = {}
            for ki in points.points_subsets_dict[ko]:  # ki is inner key
                self.shapes_subsets_dict[ko][ki] = get_shape(points.points_subsets_dict[ko][ki])


########################################################################################
# Functions mainly for BindingSites class
########################################################################################

def get_zscales_amino_acids(mol, output_log_path=None):

    """
    Get all amino acids atoms that are described by Z-scales.

    Parameters
    ----------
    mol : pandas DataFrame
        DataFrame containing atom lines from input file.
    output_log_path : string
        output_log_path: Path to log file.

    Returns
    -------
    pandas DataFrame
        DataFrame containing atom lines from input file described by Z-scales.

    """

    # Get amino acid name per row (atom)
    mol_aa = mol["subst_name"].apply(lambda x: x[0:3])

    # Get only rows (atoms) that belong to Z-scales amino acids
    mol_zscales_aa = mol[mol_aa.apply(lambda y: y in aa.zscales.index)].copy()

    # Get only rows (atoms) that DO NOT belong to Z-scales amino acids
    mol_non_zscales_aa = mol[mol_aa.apply(lambda y: y not in aa.zscales.index)].copy()

    if not mol_non_zscales_aa.empty:
        if output_log_path is not None:
            log_file = open(output_log_path, "a+")
            log_file.write("Atoms removed for binding site encoding:\n\n")
            log_file.write(mol_non_zscales_aa.to_string() + "\n\n")
            log_file.close()
        else:
            print("Atoms removed for binding site encoding:")
            print(mol_non_zscales_aa)

    return mol_zscales_aa


########################################################################################
# Functions mainly for Representatives class
########################################################################################

def get_representatives(mol, repres_key):

    """
    Extract binding site representatives.

    Parameters
    ----------
    mol : pandas DataFrame
        DataFrame containing atom lines from input file.
    repres_key : string
        repres_key: Representatives name; key in repres_dict.

    Returns
    -------
    pandas DataFrame
        DataFrame containing atom lines from input file described by Z-scales.
    """

    if repres_key == repres_keys[0]:
        return get_ca(mol)
    if repres_key == repres_keys[1]:
        return get_pca(mol)
    if repres_key == repres_keys[2]:
        return get_pc(mol)
    else:
        raise SystemExit("Unknown representative key."
                         "Select: ", ", ".join(repres_keys))


def get_ca(mol):

    """
    Extract Calpha atoms from binding site.

    Parameters
    ----------
    mol : pandas DataFrame
        DataFrame containing atom lines from input file.

    Returns
    -------
    pandas DataFrame
        DataFrame containing atom lines from input file described by Z-scales.
    """

    bs_ca = mol[mol['atom_name'] == "CA"]

    return bs_ca


def get_pca(mol):

    """
    Extract pseudocenter atoms from binding site.

    Parameters
    ----------
    mol: pandas DataFrame
        DataFrame containing atom lines from input file.

    Returns
    -------
    pandas DataFrame
        DataFrame containing atom lines from input file described by Z-scales.
    """

    # Add column containing amino acid names
    mol["amino_acid"] = [i.split("_")[0][:3] for i in mol["subst_name"]]

    # Per atom in binding site: get atoms that belong to pseudocenters
    matches = []  # Matching atoms
    pc_types = []  # Pc type of matching atoms
    pc_ids = []  # Pc ids of matching atoms
    pc_atom_ids = []  # Pc atom ids of matching atoms

    # Iterate over all atoms (lines) in binding site
    for i in mol.index:
        line = mol.loc[i]  # Atom in binding site

        # Get atoms that belong to peptide bond
        if re.search(r'^[NOC]$', line["atom_name"]):
            matches.append(True)
            if line["atom_name"] == "O":
                pc_types.append("HBA")
                pc_ids.append("PEP_HBA_1")
                pc_atom_ids.append("PEP_HBA_1_0")
            elif line["atom_name"] == "N":
                pc_types.append("HBD")
                pc_ids.append("PEP_HBD_1")
                pc_atom_ids.append("PEP_HBD_1_N")
            elif line["atom_name"] == "C":
                pc_types.append("AR")
                pc_ids.append("PEP_AR_1")
                pc_atom_ids.append("PEP_AR_1_C")

        # Get other defined atoms
        else:
            query = (line["amino_acid"] + "_" + line["atom_name"])
            matches.append(query in list(pc_atoms["pattern"]))
            if query in list(pc_atoms["pattern"]):
                ix = pc_atoms.index[pc_atoms["pattern"] == query].tolist()[0]
                pc_types.append(pc_atoms.iloc[ix]["type"])
                pc_ids.append(pc_atoms.iloc[ix]["pc_id"])
                pc_atom_ids.append(pc_atoms.iloc[ix]["pc_atom_id"])

    bs_pc_atoms = mol[matches].copy()
    bs_pc_atoms["pc_types"] = pd.Series(pc_types, index=bs_pc_atoms.index)
    bs_pc_atoms["pc_id"] = pd.Series(pc_ids, index=bs_pc_atoms.index)
    bs_pc_atoms["pc_atom_id"] = pd.Series(pc_atom_ids, index=bs_pc_atoms.index)

    return bs_pc_atoms


def get_pc(mol):

    """
    Extract pseudocenters from binding site.

    Parameters
    ----------
    mol : pandas DataFrame
        DataFrame containing atom lines from input file.

    Returns
    -------
    pandas DataFrame
        DataFrame containing atom lines from input file described by Z-scales.
    """

    # Get pseudocenter atoms
    bs_pc = get_pca(mol)

    # Loop over binding site amino acids
    for subst_name_id in set(bs_pc["subst_name"]):

        # Loop over pseudocenters of that amino acids
        for pc_id in set(bs_pc[bs_pc["subst_name"] == subst_name_id]["pc_id"]):

            # Get all rows (row indices) of binding site atoms that share the same amino acid and pseudocenter
            ix = bs_pc[(bs_pc["subst_name"] == subst_name_id) & (bs_pc["pc_id"] == pc_id)].index
            # If there is more than one atom for this pseudocenter...

            if len(ix) != 1:
                # ... calculate the mean of the corresponding atom coordinates
                bs_pc.at[ix[0], ["x", "y", "z"]] = bs_pc.loc[ix][["x", "y", "z"]].mean()
                # ... join all atom names to on string and add to dataframe in first row
                bs_pc.at[ix[0], ["atom_name"]] = ",".join(list(bs_pc.loc[ix]["atom_name"]))
                # ... remove all rows except the first (i.e. merged atoms)
                bs_pc.drop(list(ix[1:]), axis=0, inplace=True)

    # Drop pc atom ID column
    bs_pc.drop("pc_atom_id", axis=1, inplace=True)

    return bs_pc


########################################################################################
# Functions mainly for PCProperties class
########################################################################################

def get_pcproperties(repres_dict, repres_key, pcprop_key):

    """
    Extract physicochemical properties (main function).

    Parameters
    ----------
    repres_dict : Representatives.repres_dict
        Representatives stored as dictionary with several representation methods serving as key.
    repres_key : String
        Representatives name; key in repres_dict.
    pcprop_key : String
        Physicochemical property name; key in pcprop_key.

    Returns
    -------
    pandas DataFrame
        DataFrame containing physicochemical properties.
    """

    if pcprop_key == pcprop_keys[0]:
        return get_zscales(repres_dict, repres_key, 1)
    if pcprop_key == pcprop_keys[1]:
        return get_zscales(repres_dict, repres_key, 2)
    if pcprop_key == pcprop_keys[2]:
        return get_zscales(repres_dict, repres_key, 3)
    else:
        raise SystemExit('Unknown representative key.'
                         'Select: ', ", ".join(pcprop_keys))


def get_zscales(repres_dict, repres_key, z_number):

    """
    Extract Z-scales from binding site representatives.

    Parameters
    ----------
    repres_dict : Representatives.repres_dict
        Representatives stored as dictionary with several representation methods serving as key.
    repres_key : string
        Representatives name; key in repres_dict.
    z_number : int
        Number of Z-scales to be included.

    Returns
    -------
    pandas DataFrame
        DataFrame containing Z-scales.
    """

    # Get amino acid to Z-scales transformation matrix
    zs = aa.zscales

    # Transform e.g. ALA100 to ALA
    repres_aa = repres_dict[repres_key]["subst_name"].apply(lambda x: x[0:3])

    # Transform amino acids into Z-scales
    repres_zs = []
    for i in repres_aa:
        # If amino acid is in transformation matrix, add Z-scales
        if i in zs.index:
            repres_zs.append(zs.loc[i])
        # If not, terminate script and print out error message
        else:
            sys.exit("Error: Identifier " + i + " not in pre-defined Z-scales. " +
                     "Please contact dominique.sydow@charite.de")
    # Cast into DataFrame
    bs_repres_zs = pd.DataFrame(repres_zs, index=repres_dict[repres_key].index, columns=zs.columns)

    return bs_repres_zs.iloc[:, :z_number]


########################################################################################
# Functions mainly for Subsetter class
########################################################################################


def get_subset_indices(repres_dict, repres_key):

    """
    Extract feature subsets from pseudocenters (pseudocenter atoms).

    Parameters
    ----------
    repres_dict : Representatives.repres_dict
        Representatives stored as dictionary with several representation methods serving as key.
    repres_key : string
        Representatives name; key in repres_dict.

    Returns
    -------

    """

    repres = repres_dict[repres_key]
    subset = {}

    # Loop over all pseudocenter types
    for i in list(set(pc_atoms["type"])):

        # If pseudocenter type exists in dataset, save corresponding subset, else save None
        if i in set(repres["pc_types"]):
            subset[i] = repres[repres["pc_types"] == i].index
        else:
            subset[i] = None

    return subset


########################################################################################
# Functions mainly for Points class
########################################################################################

def get_points(coord_dict, pcprop_dict):

    """
    Concatenate spatial (3-dimensional) and physicochemical (N-dimensional) properties
    to an 3+N-dimensional vector for each point in dataset (i.e. representative atoms in a binding site).

    Parameters
    ----------
    coord_dict : dict of DataFrames (Coordinates.coord_dict)
        Spatial properties (=coordinates) for each representative.
        Has the same keys as Representatives.repres_dict.
    pcprop_dict : dict of dict of DataFrames (PCProperties.pcprop_dict)
        Physicochemical properties for each representative.
        Has the same top level keys as Representatives.repres_dict,
        with nested keys describing different physicochemical properties per representative type.

    Returns
    -------
    Dict of dict of DataFrames
         Spatial and physicochemical properties for each representative.
    """

    points_dict = {}

    for i in repres_keys:
        points_dict[i] = coord_dict[i]
        for j in pcprop_keys:
            points_dict[i + "_" + j] = pd.concat([coord_dict[i], pcprop_dict[i][j]], axis=1)

    return points_dict


def get_points_subsetted(points_dict, subsets_indices_dict):

    """
    Get

    Parameters
    ----------
    points_dict : dict of dict of DataFrames
        Spatial and physicochemical properties for each representative.
    subsets_indices_dict : dict of dict of DataFrames
        #FIXME

    Returns
    -------
    Dict of dict of DataFrames
        #FIXME

    """

    points_subsets_dict = {}

    for subsets_key in subsets_indices_dict.keys():
        points_keys = [key for key in points_dict.keys() if subsets_key + "_" in key]
        for points_key in points_keys:
            points_subsets_dict[points_key] = {key: points_dict[points_key].loc[value, :] for key, value in
                                               subsets_indices_dict[subsets_key].items()}

    return points_subsets_dict


########################################################################################
# Functions mainly for Shapes class
########################################################################################

def get_shape(points):

    """
    Get binding site shape with different encoding methods, depending on number of representatives (points).
    Check if

    Parameters
    ----------
    points : DataFrame
        Spatial and physicochemical properties of representatives (points).

    Returns
    -------
    Dict of dicts
        Encoded binding site information such as reference points, distances, and moments (lower level keys)
        for different encoding methods (top level keys).

    Notes
    -----
    Calculate shape if at least as many representatives are in binding site as needed references points, else return
    dictionary with key "na" and value None.

    """

    n_points = points.shape[0]
    n_dimensions = points.shape[1]

    # Select method based on number of dimensions and points
    if n_dimensions < 3:
        return {"na": None}
    if n_dimensions == 3 and n_points > 3:
        shape_3dim_dict = {"3dim_usr": get_shape_3dim_usr(points),
                           "3dim_csr": get_shape_3dim_csr(points)}
        return shape_3dim_dict
    if n_dimensions == 4 and n_points > 4:
        shape_4dim_dict = {"4_dim_electroshape": get_shape_4dim_electroshape(points)}
        return shape_4dim_dict
    if n_dimensions == 6 and n_points > 6:
        shape_6dim_dict = {"6dim": get_shape_6dim(points)}
        return shape_6dim_dict
    else:
        return {"na": None}


def get_distances_to_point(points: pd.DataFrame, ref_point: pd.Series) -> pd.Series:

    """
    Calculate distances from one point (reference point) to all other points.

    Parameters
    ----------
    points : DataFrame
        Coordinates of representatives (points) in binding site.
    ref_point : Series
        Coordinates of one reference point.

    Returns
    -------
    Series
        Distances from reference point to representatives.

    """

    distances: pd.Series = np.sqrt(((points - ref_point) ** 2).sum(axis=1))

    return distances


def get_moments(dist):

    """
    Calculate first, second, and third moment (mean, standard deviation, and skewness) for a distance distribution.

    Parameters
    ----------
    dist : Series
        Distance distribution, i.e. distances from reference point to all representatives (points)

    Returns
    -------
    Series
        First, second, and third moment of distance distribution.

    """

    # Get first, second, and third moment (mean, standard deviation, and skewness) for a distance distribution
    if len(dist) > 0:
        m1 = dist.mean()
        m2 = dist.std()
        m3 = pd.Series(cbrt(skew(dist.values, axis=0)), index=dist.columns.tolist())
    else:
        # In case there is only one data point.
        # However, this should not be possible due to restrictions in get_shape function.
        m1, m2, m3 = None, None, None

    # Store all moments in data frame
    moments = pd.concat([m1, m2, m3], axis=1)
    moments.columns = ["m1", "m2", "m3"]

    return moments


def get_closest_atom(coord):
    return coord


def get_shape_dict(ref_points, dist):

    """
    Get shape of binding site, i.e. reference points, distance distributions, and moments.

    Parameters
    ----------
    ref_points : List of Series
        Reference points (spatial and physicochemical properties).
    dist : List of Series
        Distances from each reference point to representatives (points).

    Returns
    -------
    Dict of DataFrames
        Reference points, distance distributions, and moments.

    """

    # Store reference points as data frame
    ref_points = pd.concat(ref_points, axis=1).transpose()
    ref_points.index = ["c" + str(i) for i in range(1, len(ref_points.index) + 1)]

    # Store distance distributions as data frame
    dist = pd.concat(dist, axis=1)
    dist.columns = ["dist_c" + str(i) for i in range(1, len(dist.columns) + 1)]

    # Get first, second, and third moment for each distance distribution
    moments = get_moments(dist)

    # Save shape as dictionary
    shape = {"ref_points": ref_points, "dist": dist, "moments": moments}

    return shape


def get_shape_3dim_usr(points):

    """
    Encode binding site (3-dimensional points) based on the USR method.


    Parameters
    ----------
    points : DataFrame
        Coordinates of representatives (points) in binding site.

    Returns
    -------
    dict
        Reference points, distance distributions, and moments.

    Notes
    -----
    1. Calculate reference points for a set of points:
       - c1, centroid
       - c2, closest atom to c1
       - c3, farthest atom to c1
       - c4, farthest atom to c3
    2. Calculate distances (distance distribution) from reference points to all other points.
    3. Calculate first, second, and third moment for each distance distribution.

    References
    ----------
    [1]_ Ballester and Richards, "Ultrafast shape Recognition to search compound databases for similar molecular
    shapes", J Comput Chem, 2007.

    """

    if points.shape[1] != 3:
        sys.exit(f'Error: Dimension of input (points) is not {points.shape[1]} as requested.')

    # Get centroid of input coordinates
    c1 = points.mean(axis=0)

    # Get distances from c1 to all other points
    dist_c1 = get_distances_to_point(points, c1)

    # Get closest and farthest atom to centroid c1
    c2, c3 = points.loc[dist_c1.idxmin()], points.loc[dist_c1.idxmax()]

    # Get distances from c2 to all other points, get distances from c3 to all other points
    dist_c2 = get_distances_to_point(points, c2)
    dist_c3 = get_distances_to_point(points, c3)

    # Get farthest atom to farthest atom to centroid c3
    c4 = points.loc[dist_c3.idxmax()]

    # Get distances from c4 to all other points
    dist_c4 = get_distances_to_point(points, c4)

    c = [c1, c2, c3, c4]
    dist = [dist_c1, dist_c2, dist_c3, dist_c4]

    return get_shape_dict(c, dist)


def get_shape_3dim_csr(points):

    """
    Encode binding site (3-dimensional points) based on the CSR method.


    Parameters
    ----------
    points : DataFrame
        Coordinates of representatives (points) in binding site.

    Returns
    -------
    dict
        Reference points, distance distributions, and moments.

    Notes
    -----
    1. Calculate reference points for a set of points:
       - c1, centroid
       - c2, farthest atom to c1
       - c3, farthest atom to c2
       - c4, cross product of two vectors spanning c1, c2, and c3
    2. Calculate distances (distance distribution) from reference points to all other points.
    3. Calculate first, second, and third moment for each distance distribution.

    References
    ----------
    [1]_ Armstrong et al., "Molecular similarity including chirality", J Mol Graph Mod, 2009

    """

    if points.shape[1] != 3:
        sys.exit(f'Error: Dimension of input (points) is not {points.shape[1]} as requested.')

    # Get centroid of input coordinates, and distances from c1 to all other points
    c1 = points.mean(axis=0)
    dist_c1 = get_distances_to_point(points, c1)

    # Get farthest atom to centroid c1, and distances from c2 to all other points
    c2 = points.loc[dist_c1.idxmax()]
    dist_c2 = get_distances_to_point(points, c2)

    # Get farthest atom to farthest atom to centroid c2, and distances from c3 to all other points
    c3 = points.loc[dist_c2.idxmax()]
    dist_c3 = get_distances_to_point(points, c3)

    # Get forth reference point, including chirality information
    # Define two vectors spanning c1, c2, and c3 (from c1)
    a = c2 - c1
    b = c3 - c1

    # Calculate cross product of a and b (keep same units and normalise vector to have half the norm of vector a)
    cross = np.cross(a, b)  # Cross product
    cross_norm = np.sqrt(sum(cross ** 2))  # Norm of cross product
    cross_unit = cross / cross_norm  # Cross unit vector
    a_norm = np.sqrt(sum(a ** 2))  # Norm of vector a
    c4 = pd.Series(a_norm / 2 * cross_unit, index=["x", "y", "z"])

    # Get distances from c4 to all other points
    dist_c4 = get_distances_to_point(points, c4)

    c = [c1, c2, c3, c4]
    dist = [dist_c1, dist_c2, dist_c3, dist_c4]

    return get_shape_dict(c, dist)


def get_shape_4dim_electroshape(points, scaling_factor=1):

    """
    Encode binding site (4-dimensional points) based on the ElectroShape method.


    Parameters
    ----------
    points : DataFrame
        Coordinates of representatives (points) in binding site.
    scaling_factor : int or float
        Scaling factor that can put higher or lower weight on non-spatial dimensions.

    Returns
    -------
    dict
        Reference points, distance distributions, and moments.

    Notes
    -----
    1. Calculate reference points for a set of points:
       - c1, centroid
       - c2, farthest atom to c1
       - c3, farthest atom to c2
       - c4 and c5, cross product of two vectors spanning c1, c2, and c3 with positive/negative sign in forth dimension
    2. Calculate distances (distance distribution) from reference points to all other points.
    3. Calculate first, second, and third moment for each distance distribution.

    References
    ----------
    [1]_ Armstrong et al., "ElectroShape: fast molecular similarity calculations incorporating shape, chirality and
    electrostatics", J Comput Aided Mol Des, 2010.

    """

    if points.shape[1] != 4:
        sys.exit(f'Error: Dimension of input (points) is not {points.shape[1]} as requested.')

    # Get centroid of input coordinates (in 4 dimensions), and distances from c1 to all other points
    c1 = points.mean(axis=0)
    dist_c1 = get_distances_to_point(points, c1)

    # Get farthest atom to centroid c1 (in 4 dimensions), and distances from c2 to all other points
    c2 = points.loc[dist_c1.idxmax()]
    dist_c2 = get_distances_to_point(points, c2)

    # Get farthest atom to farthest atom to centroid c2 (in 4 dimensions), and distances from c3 to all other points
    c3 = points.loc[dist_c2.idxmax()]
    dist_c3 = get_distances_to_point(points, c3)

    # Get forth and fifth reference point:
    # 1. Define two vectors spanning c1, c2, and c3 (from c1)
    a = c2 - c1
    b = c3 - c1

    # 2. Get only spatial part of a and b
    a_s = a[0:3]
    b_s = b[0:3]

    # 3. Calculate cross product of a_s and b_s
    # (keep same units and normalise vector to have half the norm of vector a)
    cross = np.cross(a_s, b_s)
    c_s = pd.Series(np.sqrt(sum(a ** 2)) / 2 * cross / (np.sqrt(sum(cross ** 2))), index=["x", "y", "z"])

    # 4. Add c to c1 to define the spatial components of the forth and fifth reference points

    # Add 4th dimension
    name_4thdim = points.columns[3]
    c = c_s.append(pd.Series([0], index=[name_4thdim]))
    c1_s = c1[0:3].append(pd.Series([0], index=[name_4thdim]))

    # Get values for 4th dimension (min and max of 4th dimension)
    max_value_4thdim = max(points.iloc[:, [3]].values)[0]
    min_value_4thdim = min(points.iloc[:, [3]].values)[0]

    c4 = c1_s + c + pd.Series([0, 0, 0, scaling_factor * max_value_4thdim], index=["x", "y", "z", name_4thdim])
    c5 = c1_s + c + pd.Series([0, 0, 0, scaling_factor * min_value_4thdim], index=["x", "y", "z", name_4thdim])

    # Get distances from c4 and c5 to all other points
    dist_c4 = get_distances_to_point(points, c4)
    dist_c5 = get_distances_to_point(points, c5)

    c = [c1, c2, c3, c4, c5]
    dist = [dist_c1, dist_c2, dist_c3, dist_c4, dist_c5]

    return get_shape_dict(c, dist)


def get_shape_6dim(points, scaling_factor=1):

    """
    Encode binding site in 6D.

    Parameters
    ----------
    points : DataFrame
        Coordinates of representatives (points) in binding site.
    scaling_factor : int or float
        Scaling factor that can put higher or lower weight on non-spatial dimensions.

    Returns
    -------


    Notes
    -----
    1. Calculate reference points for a set of points:
       - c1, centroid
       - c2, closest atom to c1
       - c3, farthest atom to c1
       - c4, farthest atom to c3
       - c5, nearest atom to translated and scaled cross product of two vectors spanning c1, c2, and c3
       - c6, nearest atom to translated and scaled cross product of two vectors spanning c1, c3, and c4
       - c7, nearest atom to translated and scaled cross product of two vectors spanning c1, c4, and c2
    2. Calculate distances (distance distribution) from reference points to all other points.
    3. Calculate first, second, and third moment for each distance distribution.

    """

    if points.shape[1] != 6:
        sys.exit(f'Error: Dimension of input (points) is not {points.shape[1]} as requested.')

    # Get centroid of input coordinates (in 7 dimensions), and distances from c1 to all other points
    c1 = points.mean(axis=0)
    dist_c1 = get_distances_to_point(points, c1)

    # Get closest and farthest atom to centroid c1 (in 7 dimensions),
    # and distances from c2 and c3 to all other points
    c2, c3 = points.loc[dist_c1.idxmin()], points.loc[dist_c1.idxmax()]
    dist_c2 = get_distances_to_point(points, c2)
    dist_c3 = get_distances_to_point(points, c3)

    # Get farthest atom to farthest atom to centroid c2 (in 7 dimensions),
    # and distances from c3 to all other points
    c4 = points.loc[dist_c3.idxmax()]
    dist_c4 = get_distances_to_point(points, c4)

    # Get adjusted cross product
    c5_3d = get_adjusted_3d_cross_product(c1, c2, c3)  # FIXME order of importance, right?
    c6_3d = get_adjusted_3d_cross_product(c1, c3, c4)
    c7_3d = get_adjusted_3d_cross_product(c1, c4, c2)

    # Get remaining reference points as nearest atoms to adjusted cross products
    c5 = get_nearest_atom(c5_3d, points, scaling_factor)
    c6 = get_nearest_atom(c6_3d, points, scaling_factor)
    c7 = get_nearest_atom(c7_3d, points, scaling_factor)

    # Get distances from c5, c6, and c7 to all other points
    dist_c5 = get_distances_to_point(points, c5)
    dist_c6 = get_distances_to_point(points, c6)
    dist_c7 = get_distances_to_point(points, c7)

    c = [c1, c2, c3, c4, c5, c6, c7]
    dist = [dist_c1, dist_c2, dist_c3, dist_c4, dist_c5, dist_c6, dist_c7]

    return get_shape_dict(c, dist)


def get_adjusted_3d_cross_product(coord_origin, coord_point_a, coord_point_b):

    """
    Calculates a translated and scaled 3D cross product vector based on three input vectors.

    Parameters
    ----------
    coord_origin : Series
        Point in 6D as coordinate system (part of cross product).
    coord_point_a : Series
        Point in 6D (part of cross product).
    coord_point_b : Series
        Point in 6D (part of cross product).

    Returns
    -------
    Series
        Translated and scaled cross product 3D.

    Notes
    -----
    1. Generate two vectors based on the first three coordinates of the three input vectors (origin, point_a, and
       point_b), originating from origin.
    2. Calculate their cross product
       - scaled to length of the mean of both vectors and
       - translated to the origin.
    3. Get nearest point in points (in 3D) and return its 6D vector.

    """

    if not coord_origin.size == coord_point_a.size == coord_point_b.size:
        sys.exit('Error: The three input Series are not of same length.')
    if not coord_origin.size > 2:
        sys.exit('Error: The three input Series are not at least of length 3.')

    # Span vectors to point a and b from origin point
    a = coord_point_a - coord_origin
    b = coord_point_b - coord_origin

    # Get only first three dimensions
    a_3d = a[0:3]
    b_3d = b[0:3]

    # Calculate norm of vectors and their mean norm
    a_3d_norm = np.sqrt(sum(a_3d ** 2))
    b_3d_norm = np.sqrt(sum(b_3d ** 2))
    ab_3d_norm = (a_3d_norm + b_3d_norm) / 2

    # Calculate cross product
    cross = np.cross(a_3d, b_3d)

    # Calculate norm of cross product
    cross_norm = np.sqrt(sum(cross ** 2))

    # Calculate unit vector of cross product
    cross_unit = cross / cross_norm

    # Adjust cross product to length of the mean of both vectors described by the cross product
    cross_adjusted = ab_3d_norm * cross_unit

    # Cast adjusted cross product to Series
    coord_cross = pd.Series(cross_adjusted, index=a_3d.index)

    # Move adjusted cross product so that it originates from origin point
    c_3d = coord_origin[0:3] + coord_cross

    return c_3d


def get_nearest_atom(point, points, scaling_factor):

    """
    Get the point (N-dimensional) in a set of points that is nearest (in 3D) to an input point (3D).

    Parameters
    ----------
    point : Series
        Point in 3D
    points : DataFrame
        Points in at least 3D.

    Returns
    -------
    Series
        Point in **points** with all dimensions that is nearest in 3D to **point**.

    """

    if not point.size == 3:
        sys.exit('Error: Input point is not of length 3.')
    if not points.shape[1] > 2:
        sys.exit('Error: Input points are not of at least length 3.')

    # Get distances (in space, i.e. 3D) to all points
    dist_c_3d = get_distances_to_point(points.iloc[:, 0:3], point)

    # Get nearest atom (in space, i.e. 3D) and save its 6D vector
    c_6d = points.loc[dist_c_3d.idxmin()]  # FIXME loc or iloc

    # Apply scaling factor on non-spatial dimensions
    scaling_vector = pd.Series([1] * 3 + [scaling_factor] * (points.shape[1]-3), index=c_6d.index)
    c_6d = c_6d * scaling_vector

    return c_6d


def get_adjusted_6d_cross_product():
    pass
