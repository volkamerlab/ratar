"""
cli.py

Read-across the targetome -
An integrated structure- and ligand-based workbench for computational target prediction and novel tool compound design

Handles the CLI functions for processing the encoding of multiple binding sites.
"""

import argparse
import datetime
import logging
import logging.config
from pathlib import Path
import subprocess
import platform

from .auxiliary import create_directory
from .encoding import process_encoding


def main():
    """
    Parse the arguments given when calling this script.

    Returns
    -------
    argparse.Namespace
        Input arguments, i.e. molecule structure file path and output directory.
    """

    # Parse arguments
    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # Subcommand encode
    encode_subparser = subparsers.add_parser("encode")
    encode_subparser.add_argument(
        "-i", "--input_mol_path", help="Path to molecule structure file(s).", required=True
    )
    encode_subparser.add_argument(
        "-o", "--output_dir", help="Path to output directory.", required=True
    )
    encode_subparser.set_defaults(func=main_encode)

    # Subcommand compare
    compare_subparser = subparsers.add_parser("compare")
    compare_subparser.set_defaults(func=main_compare)

    # Set as variables
    args = parser.parse_args()
    try:
        args.func(args)
    except AttributeError:
        # Run help if no arguments were given
        subprocess.run(["ratar", "-h"])


def configure_logger(filename=None, level=logging.INFO):
    """
    Configure logging.
    Parameters
    ----------
    filename : str or None
        Path to log file.
    level : int
        Logging level for ratar package (default: INFO).
    """

    # Create custom logger
    logger = logging.getLogger(__name__)
    # Set logger levels
    logger.setLevel(level)
    # Get formatter
    formatter = logging.Formatter(logging.BASIC_FORMAT)
    # Set a stream and a file handler
    # Create handlers
    s_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(filename=Path(filename) / "ratar.log")
    # Set formatting for these handlers
    s_handler.setFormatter(formatter)
    # Add both handlers to both loggers
    logger.addHandler(s_handler)
    logger.addHandler(f_handler)

    # Set up file handler if
    # - log file is given
    # - we are not under Windows, since logging and multiprocessing do not work here
    #   see more details here: https://github.com/volkamerlab/kissim/pull/49
    if filename and platform.system() != "Windows":
        filename = Path(filename)
        f_handler = logging.FileHandler(filename.parent / f"{filename.stem}.log", mode="w")
        f_handler.setFormatter(formatter)
        logger.addHandler(f_handler)

    return logger


def main_encode(args):
    """
    Encode structures.

    Parameters
    ----------
    args : argsparse.Namespace
        CLI arguments.
    """

    # Get start time of script
    encoding_start = datetime.datetime.now()

    # Create output folder
    create_directory(args.output_dir)

    logger = configure_logger(args.output_dir)

    # Log IO
    logger.info("IO", extra={"molecule_id": "all"})
    logger.info(f"Input: {args.input_mol_path}", extra={"molecule_id": "all"})
    logger.info(f"Output: {args.output_dir}", extra={"molecule_id": "all"})

    # Process encoding
    logger.info(f"PROCESS ENCODING...", extra={"molecule_id": "all"})
    process_encoding(args.input_mol_path, args.output_dir, remove_solvent=True)

    # Get end time of encoding step and runtime
    encoding_end = datetime.datetime.now()
    encoding_runtime = encoding_end - encoding_start

    # Log runtime
    logger.info(f"RUNTIME", extra={"molecule_id": "all"})
    logger.info(f"Encoding step: {encoding_runtime}", extra={"molecule_id": "all"})


def main_compare(args):
    """
    Compare structures.

    Parameters
    ----------
    args : argsparse.Namespace
        CLI arguments.
    """

    raise NotImplementedError(f"Calculating similarities from the CLI is not implemented, yet.")
