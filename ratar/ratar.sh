#!/bin/bash

########################################################################################
# Binding site encoding - main bash script
########################################################################################

data_dir="/home/dominique/Documents/data"
output_dir="/home/dominique/Documents/projects/ratar-data/results"


########################################################################################
# One mol2 file (possibly with multiple entries)
########################################################################################

# Data set scPDB-EFGR
#python ratar_encoding.py -i ${data_dir}"/scpdb/egfr_20180807/scPDB_20190128_egfr.mol2" -o ${output_dir}


########################################################################################
# Multiple mol2 files
########################################################################################

# Full scPDB
#python ratar_encoding.py -i ${data_dir}"/scpdb/full_20180807/*/site.mol2" -o ${output_dir}

# Subset of scPDB
python ratar_encoding.py -i ${data_dir}"/scpdb/test_20180807/*/site.mol2" -o ${output_dir}"/scpdb/test_20180807/"

# Data set scPDB sent by Didier Rognan
#python ratar_encoding.py -i ${data_dir}"/benchmarking/fuzcav/sim_dis_pairs/structures/*/site_CA_Met_pp.mol2" -o ${output_dir}"/benchmarking/fuzcav/sim_dis_pairs/"
