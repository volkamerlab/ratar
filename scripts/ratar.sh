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
ratar -i ${data_dir}"/scpdb/egfr_20190128/scPDB_20190128_egfr.mol2" -o ${output_dir}"/scpdb/egfr_20190128/"


########################################################################################
# Multiple mol2 files
########################################################################################

# Full scPDB
ratar -i ${data_dir}"/scpdb/full_20180807/*/site.mol2" -o ${output_dir}

# Subset of scPDB
ratar -i ${data_dir}"/scpdb/test_20180807/*/site.mol2" -o ${output_dir}"/scpdb/test_20180807/"

# Benchmarking dataset - FuzCav
ratar -i ${data_dir}"/benchmarking/fuzcav/sim_dis_pairs/structures/*/site_CA_Met.mol2" -o ${output_dir}"/benchmarking/fuzcav/sim_dis_pairs/"

# Benchmarking dataset - TOUGH-M1
ratar -i ${data_dir}"/benchmarking/TOUGH-M1/structures/*/site_CA_Met.mol2" -o ${output_dir}"/benchmarking/fuzcav/sim_dis_pairs/"

# KLIFS test
ratar -i ${data_dir}"/klifs/test_20190506/structures/HUMAN/*/*/pocket.mol2" -o ${output_dir}"/klifs/test_20190506/HUMAN/"

# KLIFS full
ratar -i ${data_dir}"/klifs/klifs_20190506/structures/HUMAN/*/*/pocket.mol2" -o ${output_dir}"/klifs/klifs_20190506/HUMAN/"


########################################################################################
# Multiple pdb files
########################################################################################

# TBA