"""
core.py

Read-across the targetome -
An integrated structure- and ligand-based workbench for computational target prediction and novel tool compound design

Handles the CLI functions for processing the encoding of multiple binding sites.
"""

import argparse
import datetime
from pathlib import Path

from .auxiliary import create_directory
from .encoding import process_encoding


def parse_arguments():
    """
    Parse the arguments given when calling this script.

    :return: Input molecule structure file path and output directory.
    :rtype: Strings
    """

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_mol_path', help='Path to molecule structure file(s).', required=True)
    parser.add_argument('-o', '--output_dir', help='Path to output directory.', required=True)

    # Set as variables
    args = parser.parse_args()
    input_mol_path = args.input_mol_path
    output_dir = args.output_dir

    print(f'Input: {input_mol_path}')
    print(f'Output: {output_dir}')

    return input_mol_path, output_dir


def main():
    # Get start time of script
    encoding_start = datetime.datetime.now()

    # Parse arguments
    input_mol_path, output_dir = parse_arguments()

    # Create output folder
    create_directory(output_dir)

    # Log IO files
    log_file = open(Path(output_dir) / 'ratar.log', 'w')
    log_file.write(f'------------------------------------------------------------\n')
    log_file.write(f'IO\n')
    log_file.write(f'------------------------------------------------------------\n\n')
    log_file.write(f'Input: {input_mol_path}\n')

    # Log encoding step processing
    log_file.write(f'Output: {output_dir}\n\n')
    log_file.write(f'------------------------------------------------------------\n')
    log_file.write(f'PROCESS ENCODING\n')
    log_file.write(f'------------------------------------------------------------\n\n')
    log_file.close()

    # Process encoding
    process_encoding(input_mol_path, output_dir)

    # Get end time of encoding step and runtime
    encoding_end = datetime.datetime.now()
    encoding_runtime = encoding_end - encoding_start

    # Log runtime
    log_file = open(Path(output_dir) / 'ratar.log', 'a+')
    log_file.write(f'\n------------------------------------------------------------\n')
    log_file.write(f'RUNTIME\n')
    log_file.write(f'------------------------------------------------------------\n\n')
    log_file.write(f'Encoding step: {encoding_runtime}\n')
    log_file.close()


if __name__ == '__main__':
    main()