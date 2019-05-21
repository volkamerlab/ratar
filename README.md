ratar
==============================

[//]: # (Badges)
[![Travis Build Status](https://travis-ci.org/REPLACE_WITH_OWNER_ACCOUNT/ratar.png)](https://travis-ci.org/REPLACE_WITH_OWNER_ACCOUNT/ratar)
[![AppVeyor Build status](https://ci.appveyor.com/api/projects/status/REPLACE_WITH_APPVEYOR_LINK/branch/master?svg=true)](https://ci.appveyor.com/project/REPLACE_WITH_OWNER_ACCOUNT/ratar/branch/master)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/ratar/branch/master/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/ratar/branch/master)

### Project descripition

**Read-across the targetome - an integrated structure- and ligand-based workbench for computational target prediction and novel tool compound design**

![Ratar overview](https://github.com/dominiquesydow/ratar/blob/master/README_files/fig_ratar_overview.png)

How to probe and validate a potential pathway or target remains one of the key questions in basic research in life sciences. Often these investigations lack suitable chemical tool compounds for the elucidation of the function of a specific protein. Therefore, large consortia, like the Structural Genomics Consortium, have formed to generate tool compounds for the validation of biological targets via classical chemical synthesis and extensive protein structure determination efforts. While these consortia will continue with their focused experimental approaches, the read-across the targetome project will offer a computational solution for the generation of a comprehensive set of tool compounds for novel targets. The project hypothesis is based on the similarity principle (similar pockets bind similar compounds) with the ultimate goal of using protein pocket similarity to extrapolate compound information from one target to another. With this concept, the investigator seeks to answer the central question: Can binding site similarity be used to propose tool compounds for novel targets? Based on this new paradigm for tool compound identification, a holistic workbench will be developed. With currently over 127,000 macromolecular structures in the protein data bank (PDB) and over 1.68 million tested compounds in ChEMBL, a wealth of structural and binding data is freely available, which will be systematically explored here. First, protein structures will be collected from the PDB. For all structures per protein, potential pockets will be identified and clustered to ensemble pockets. Next, a novel efficient structure-based binding site comparison algorithm will be developed to find the most similar pockets considering protein flexibility in terms of ensembles. Ligand and binding data will be extracted from ChEMBL, filtered and assigned to the respective pockets. Using the novel algorithm, ligands known to bind to the detected neighboring pockets can be elucidated. These compounds can be selected as chemical probes for functional annotations or as novel focused compound libraries for virtual screening. The whole procedure will be made available as a web-service to combine the data and methods in one place, as well as to allow simplified access to the methodology for researchers from different life science disciplines.The combined usage of available structural and chemical information, as well as knowledge transfer from known compounds to novel similar binding pockets, will help to speed-up biological and pharmaceutical research in the area of pathway and target validation. 

### Copyright

Copyright (c) 2019, Dominique Sydow


### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.0.
