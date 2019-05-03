"""
ratar.py

Read-across the targetome -
An integrated structure- and ligand-based workbench for computational target prediction and novel tool compound design

Handles the primary functions for processing the encoding of multiple binding sites.
"""


from auxiliary import *
from encoding import encode_binding_site, save_binding_site, save_cgo_file

import datetime
import argparse
import glob


def parse_arguments():
    """
    Parse the arguments given when calling this script.

    :return: Input mol2 file path and output directory.
    :rtype: Strings
    """

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_mol2_path", help="Path to mol2 file(s).", required=True)
    parser.add_argument("-o", "--output_dir", help="Path to output directory.", required=True)

    # Set as variables
    args = parser.parse_args()
    input_mol2_path = args.input_mol2_path
    output_dir = args.output_dir

    print(f'Input: {input_mol2_path}')
    print(f'Output: {output_dir}')

    return input_mol2_path, output_dir


def process_encoding(input_mol2_path, output_dir):
    """
    Process a list of mol2 files (retrieved by an input path to one or multiple files) and
    save per binding site multiple output files to an output directory.

    Each binding site is processed as follows:
      * Create all necessary output directories and sets all necessary file paths.
      * Encode the binding site.
      * Save the encoded binding sites as pickle file (alongside a log file).
      * Save the reference points as PyMol cgo file.

    The output file systems is constructed as follows:

    output_dir/
      encoding/
        pdb_id_1/
          ratar_encoding.p
          ratar_encoding.log
          ref_points_cgo.py
        pdb_id_2/
          ...
      ratar.log


    :param input_mol2_path: Path to mol2 file(s), can include a wildcard to match multiple files.
    :type input_mol2_path: String

    :param output_dir: Output directory.
    :type output_dir: String

    :return: No return value.
    :rtype: None
    """

    # Get all mol2 files
    input_mol2_path_list = glob.glob(input_mol2_path)
    input_mol2_path_list = input_mol2_path_list

    # Get number of mol2 files and set mol2 counter
    mol2_sum = len(input_mol2_path_list)
    mol2_counter = 0

    # Iterate over all binding sites (mol2 files)
    for mol2 in input_mol2_path_list:

        # Increment mol2 counter
        mol2_counter = mol2_counter + 1

        # Load binding site from mol2 file
        bs_loader = MolFileLoader(mol2)
        pmols = bs_loader.pmols

        # Get number of pmol objects and set pmol counter
        pmol_sum = len(pmols)
        pmol_counter = 0

        # Iterate over all binding sites in mol2 file
        for pmol in pmols:

            # Increment pmol counter
            pmol_counter = pmol_counter + 1

            # Get iteration progress
            progress_string = f'{mol2_counter}/{mol2_sum} mol2 files - {pmol_counter}/{pmol_sum} pmol objects: {pmol.code}'

            # Print iteration process
            print(progress_string)

            # Log iteration process
            log_file = open(output_dir + "/ratar.log", "a+")
            log_file.write(f'{progress_string}\n')
            log_file.close()

            # Process single binding site:

            # Create output folder
            pdb_id_encoding = output_dir + "/encoding/" + pmol.code
            create_folder(pdb_id_encoding)

            # Get output file paths
            output_log_path = pdb_id_encoding + "/ratar_encoding.log"
            output_enc_path = pdb_id_encoding + "/ratar_encoding.p"
            output_cgo_path = pdb_id_encoding + "/ref_points_cgo.py"

            # Encode binding site
            binding_site = encode_binding_site(pmol, output_log_path)

            # Save binding site
            save_binding_site(binding_site, output_enc_path)

            # Save binding site reference points as cgo file
            save_cgo_file(binding_site, output_cgo_path)


if __name__ == "__main__":

    # Get start time of script
    encoding_start = datetime.datetime.now()

    # Parse arguments
    input_mol2_path, output_dir = parse_arguments()

    # Create output folder
    create_folder(output_dir)

    # Log IO files
    log_file = open(output_dir + "/ratar.log", "w")
    log_file.write(f'------------------------------------------------------------\n')
    log_file.write(f'IO\n')
    log_file.write(f'------------------------------------------------------------\n\n')
    log_file.write(f'Input: {input_mol2_path}\n')

    # Log encoding step processing
    log_file.write(f'Output: {output_dir}\n\n')
    log_file.write(f'------------------------------------------------------------\n')
    log_file.write(f'PROCESS ENCODING\n')
    log_file.write(f'------------------------------------------------------------\n\n')
    log_file.close()

    # Process encoding
    process_encoding(input_mol2_path, output_dir)

    # Get end time of encoding step and runtime
    encoding_end = datetime.datetime.now()
    encoding_runtime = encoding_end - encoding_start

    # Log runtime
    log_file = open(output_dir + "/ratar.log", "a+")
    log_file.write(f'\n------------------------------------------------------------\n')
    log_file.write(f'RUNTIME\n')
    log_file.write(f'------------------------------------------------------------\n\n')
    log_file.write(f'Encoding step: {encoding_runtime}\n"')
    log_file.close()
