########################################################################################
# Binding site encoding - main script
########################################################################################

# This script runs the binding site encoding step.


########################################################################################
# Import modules
########################################################################################

from auxiliary import *
from encoding import encode_binding_site, save_binding_site, save_cgo_file

import datetime
import argparse
import sys
import glob

# Print absolute path of the executable binary for the Python interpreter
print(sys.executable)

# Get start time of script
script_start = datetime.datetime.now()


########################################################################################
# Parse bash arguments to variables
########################################################################################

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_mol2_path", help="Path to mol2 file(s).",
                    required=True)
parser.add_argument("-o", "--output_dir", help="Path to output directory.",
                    required=True)

# Set as variables
args = parser.parse_args()
input_mol2_path = args.input_mol2_path
output_dir = args.output_dir

print("Input: %s" % input_mol2_path)
print("Output: %s" % output_dir)


########################################################################################
# Define file paths
########################################################################################

# Get input name (data name); fetches the subfolder name in data folder
input_split = input_mol2_path.split(sep="/")

# Set output files
output_dir = output_dir + "/results/encoding/" + \
             "/".join(input_split[input_split.index("data")+1:input_split.index("data")+2+1])
print(output_dir)
output_log_path = output_dir + "/binding_sites.log"

# Create output folder
create_folder(output_dir)
create_folder(output_dir + "/binding_sites")
create_folder(output_dir + "/cgo_files")


########################################################################################
# Process binding sites
########################################################################################

# Log IO files
log_file = open(output_dir + "/binding_sites.log", "w")
log_file.write("------------------------------------------------------------\n")
log_file.write("IO\n")
log_file.write("------------------------------------------------------------\n\n")
log_file.write("Input: " + input_mol2_path + "\n")
log_file.write("Output: " + output_dir + "\n\n")
log_file.close()

# Get all mol2 files
input_mol2_path_list = glob.glob(input_mol2_path)
input_mol2_path_list = input_mol2_path_list

# Log start of binding site processing
log_file = open(output_log_path, "a+")
log_file.write("------------------------------------------------------------\n")
log_file.write("PROCESS BINDING SITES\n")
log_file.write("------------------------------------------------------------\n\n")
log_file.close()

# Get number of mol2 files and set mol2 counter
mol2_sum = len(input_mol2_path_list)
mol2_counter = 0

# Iterate over all binding sites (mol2 files)
for mol2 in input_mol2_path_list:

    # Increment mol2 counter
    mol2_counter = mol2_counter + 1

    # Load binding site from mol2 file
    bs_loader = Mol2Loader(mol2, output_log_path)
    pmols = bs_loader.pmols

    # Get number of pmol objects and set pmol counter
    pmol_sum = len(pmols)
    pmol_counter = 0

    # Iterate over all binding sites in mol2 file
    for pmol in pmols:

        # Increment pmol counter
        pmol_counter = pmol_counter + 1

        # Print out iteration progress
        progress_string = str(mol2_counter) + "/" + str(mol2_sum) + " mol2 files - " + \
                          str(pmol_counter) + "/" + str(pmol_sum) + " pmol objects: " + pmol.code
        print(progress_string)

        # Add binding site information to log file
        if output_log_path is not None:
            log_file = open(output_log_path, "a+")
            log_file.write("\n" + progress_string + "\n")
            log_file.write("------------------------------------------------------------\n\n")
            log_file.close()

        # Encode binding site
        binding_site = encode_binding_site(pmol, output_log_path)

        # Save binding site
        save_binding_site(binding_site, output_dir)

        # Save binding site reference points as cgo file
        save_cgo_file(binding_site, output_dir)


# Add run time to log file
script_end = datetime.datetime.now()
runtime = script_end - script_start

log_file = open(output_log_path, "a+")
log_file.write("------------------------------------------------------------\n")
log_file.write("RUNTIME\n")
log_file.write("------------------------------------------------------------\n\n")
log_file.write("Run time: " + str(runtime))
log_file.close()
