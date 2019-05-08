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
#python ratar.py -i ${data_dir}"/scpdb/egfr_20180807/scPDB_20190128_egfr.mol2" -o ${output_dir}


########################################################################################
# Multiple mol2 files
########################################################################################

# Full scPDB
#python ratar.py -i ${data_dir}"/scpdb/full_20180807/*/site.mol2" -o ${output_dir}

# Subset of scPDB
#python ratar.py -i ${data_dir}"/scpdb/test_20180807/*/site.mol2" -o ${output_dir}"/scpdb/test_20180807/"

# Benchmarking dataset - FuzCav
#python ratar.py -i ${data_dir}"/benchmarking/fuzcav/sim_dis_pairs/structures/*/site_CA_Met_pp.mol2" -o ${output_dir}"/benchmarking/fuzcav/sim_dis_pairs/"

# Benchmarking dataset - TOUGH-M1
#python ratar.py -i ${data_dir}"/benchmarking/TOUGH-M1/structures/*/site_CA_Met_pp.mol2" -o ${output_dir}"/benchmarking/fuzcav/sim_dis_pairs/"

# KLIFS EGFR
#python ratar.py -i ${data_dir}"/klifs/egfr_20190506/structures/HUMAN/EGFR/*/pocket_pp.mol2" -o ${output_dir}"/klifs/egfr_20190509/HUMAN/EGRF/"

# KLIFS full
#cd /home/dominique/Documents/klifs/klifs_20190506/structures/HUMAN
#for i in *; do for j in $i"/"*; do echo $j"/pocket.mol2"; done; done
#for i in *; do for j in $i"/"*; do less $j"/pocket.mol2" | awk '!($10="")' > $j"/pocket_pp.mol2"; done; done
python ratar.py -i ${data_dir}"/klifs/klifs_20190506/structures/HUMAN/*/*/pocket_pp.mol2" -o ${output_dir}"/klifs/klifs_20190506/HUMAN/"