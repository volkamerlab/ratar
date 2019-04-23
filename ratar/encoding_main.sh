#!/bin/bash

########################################################################################
# Binding site encoding - main bash script
########################################################################################

data_dir="/home/dominique/Documents/data"
output_dir="/home/dominique/Documents/projects/ratar-data"


########################################################################################
# One mol2 file (possibly with multiple entries)
########################################################################################

# Data set scPDB-EFGR
#python encoding_main.py -i ${data_dir}"/data/scpdb/egfr_20180807/scPDB_20190128_egfr.mol2" -o ${output_dir}


########################################################################################
# Multiple mol2 files
########################################################################################

# Full scPDB
#python encoding_main.py -i ${data_dir}"/data/scpdb/full_20180807/*/site.mol2" -o ${output_dir}

# Subset of scPDB
python encoding_main.py -i ${data_dir}"/scpdb/test_20180807/*/site.mol2" -o ${output_dir}


# Data set scPDB sent by Didier Rognan
#python encoding_main.py -i ${data_dir}"/data/benchmarking/fuzcav/sim_dis_pairs/*/site_CA_Met_pp.mol2" -o ${output_dir}
