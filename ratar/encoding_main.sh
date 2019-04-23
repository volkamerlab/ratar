#!/bin/bash

########################################################################################
# Binding site encoding - main bash script
########################################################################################

project_path="/home/dominique/Documents/projects/readacross_targetome"


########################################################################################
# One mol2 file (possibly with multiple entries)
########################################################################################

# Data set scPDB-EFGR
#python encoding_main.py -i $project_path"/data/scPDB_20190128_egfr/scPDB_20190128_egfr.mol2"


########################################################################################
# Multiple mol2 files
########################################################################################

# Complete scPDB
python encoding_main.py -i $project_path"/data/scPDB_20180807/*/site.mol2"

# Subset of scPDB
#python encoding_main.py -i $project_path"/data/scPDB_20180807_test/*/site.mol2"


# Data set scPDB sent by Didier Rognan
#python encoding_main.py -i $project_path"/data/scPDB_rognan/*/site_CA_Met_pp.mol2"
