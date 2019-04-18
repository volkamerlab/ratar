"""
ratar
Read-across the targetome - an integrated structure- and ligand-based workbench for computational target prediction and novel tool compound design
"""

# Add imports here
from .ratar import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
