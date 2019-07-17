"""
core.py

Read-across the targetome -
An integrated structure- and ligand-based workbench for computational target prediction and novel tool compound design

Handles the CLI functions for processing the encoding of multiple binding sites.
"""

import argparse
import datetime
import logging
import logging.config
from pathlib import Path

from .auxiliary import create_directory
from .encoding import process_encoding


def parse_arguments():
    """
    Parse the arguments given when calling this script.

    Returns
    -------
    argparse.Namespace
        Input arguments, i.e. molecule structure file path and output directory.
    """

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_mol_path', help='Path to molecule structure file(s).', required=True)
    parser.add_argument('-o', '--output_dir', help='Path to output directory.', required=True)

    # Set as variables
    args = parser.parse_args()

    return args


def main():
    """
    Main ratar function to process one or multiple binding sites.
    """

    # Get start time of script
    encoding_start = datetime.datetime.now()

    # Parse arguments
    args = parse_arguments()

    # Create output folder
    create_directory(args.output_dir)

    # Create custom logger
    logging.config.fileConfig('logging.conf', defaults={'logfilename': str(Path(args.output_dir) / 'ratar.log')})

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(filename=Path(args.output_dir) / 'ratar.log')

    # Get handler
    logger = logging.getLogger(__name__)

    # Add handler to logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    # Log IO
    logger.info('IO', extra={'molecule_id': 'all'})
    logger.info(f'Input: {args.input_mol_path}', extra={'molecule_id': 'all'})
    logger.info(f'Output: {args.output_dir}', extra={'molecule_id': 'all'})

    # Process encoding
    logger.info(f'PROCESS ENCODING...', extra={'molecule_id': 'all'})
    process_encoding(args.input_mol_path, args.output_dir, remove_solvent=True)

    # Get end time of encoding step and runtime
    encoding_end = datetime.datetime.now()
    encoding_runtime = encoding_end - encoding_start

    # Log runtime
    logger.info(f'RUNTIME', extra={'molecule_id': 'all'})
    logger.info(f'Encoding step: {encoding_runtime}', extra={'molecule_id': 'all'})


if __name__ == '__main__':
    main()
