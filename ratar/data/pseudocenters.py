########################################################################################
# Pre-processing script for pseudocenter data (Schmitt et al. 2002)
########################################################################################

# This script processes the pseudocenters csv script.

import pandas as pd
import pickle


########################################################################################
# Global variables
########################################################################################

# Path to readacross_targetome folder
project_path = "/home/dominique/Documents/projects/readacross_targetome"


########################################################################################
# Pseudocenters
########################################################################################

# Load pseudocenter text file
pc_df = pd.read_csv(project_path + "/data/pseudocenters.csv", header=None)
pc_df.columns = ["amino_acid", "pc_type", "origin_atoms"]

# Change case of amino acid column
pc_df["amino_acid"] = [i.upper() for i in pc_df["amino_acid"]]

# Format column with multiple entries
pc_df["origin_atoms"] = [i.split(" ") for i in pc_df["origin_atoms"]]

# Add pseudocenter IDs
# Why? Some amino acids have several features of one features type, e.g. ARG has three HBD features).

pc_ids = []

# Initialize variables
amino_acid = ""
pc_type = ""
id_prefix_old = ""
id_suffix = 1

for i in pc_df.index:
    # Read dataframe line by line
    line = pc_df.iloc[i]

    # Get amino acid and feature type
    amino_acid = line["amino_acid"]
    pc_type = line["pc_type"]

    # Create prefix of pseudocenter ID
    id_prefix_new = amino_acid + "_" + pc_type

    # Now add suffix to pseudocenter ID
    # If prefix was not seen before, add suffix 1
    if id_prefix_new != id_prefix_old:
        id_suffix = 1
        pc_ids.append(id_prefix_new + "_" + str(id_suffix))
    # If prefix was seen before, increment and add suffix
    else:
        id_suffix = id_suffix + 1
        pc_ids.append(id_prefix_new + "_" + str(id_suffix))

    # Update prefix
    id_prefix_old = id_prefix_new

# Add pseudocenter IDs to dataframe
pc_df.insert(loc=0, column='pc_id', value=pc_ids)

# Save pseudocenter dataframe to pickle file
pickle.dump(pc_df, open(project_path + "/data/pseudocenters.p", "wb"))


########################################################################################
# Pseudocenter atoms
########################################################################################

# Define a list of pseudocenter atoms
pc_atom_pattern = []  # To be matched against binding site atoms in encoding.py: aa_atom pattern
pc_atom_types = []  # Pseudocenter (= pseudocenter atom) type
pc_ids = []  # Pseudocenter ID
pc_atom_ids = []  # Pseudocenter atom ID

for i in pc_df.index:
    line = pc_df.iloc[i]
    for j in line["origin_atoms"]:  # Some pseudocenters consist of several atoms
        pc_atom_pattern.append(line["amino_acid"] + "_" + j)
        pc_atom_types.append(line["pc_type"])
        pc_ids.append(line["pc_id"])
        pc_atom_ids.append(line["pc_id"] + "_" + j)

# Save to dict
pc_per_atom = {"pc_atom_id": pc_atom_ids,
               "pc_id": pc_ids,
               "pattern": pc_atom_pattern,
               "type": pc_atom_types}

# Transform dict to df
pc_per_atom = pd.DataFrame.from_dict(pc_per_atom)

# Save pseudocenter atoms to pickle file
pickle.dump(pc_per_atom, open(project_path + "/data/pseudocenter_atoms.p", "wb"))
