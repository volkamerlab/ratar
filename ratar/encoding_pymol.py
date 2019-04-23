import glob
import pickle

from pymol import cmd

from biopandas.mol2 import PandasMol2
from biopandas.mol2 import split_multimol2


project_path = "/home/dominique/Documents/projects/readacross_targetome"

########################################################################################################################
# Pymol executable functions
########################################################################################################################


def load_scpdb_files(pdb_list):
    """

    :param pdb_list:
    :return:
    """

    # Reinitialize pymol
    cmd.reinitialize()

    # Set path scPDB data
    data_name = "scPDB_20180807_test"
    mol2_dir = project_path + "/data/" + data_name
    cgo_dir = project_path + "/results/encoding/" + data_name + "/cgo_files"

    # Reinitialize pymol
    cmd.reinitialize()

    for pdb in pdb_list:
        print(pdb)

        # Check if PDB in scPDB dataset
        if not glob.glob(mol2_dir + "/" + pdb + "*/"):
            print("Error: PDB ID is not in scPDB dataset.")

        else:
            print("Load mol2 files...")
            # Load files
            # files = ["protein.mol2", "cavityALL.mol2", "cavity6.mol2", "site.mol2", "ligand.mol2", "ints_M.mol2"]
            files = ["protein.mol2", "site.mol2", "ligand.mol2"]

            for file in files:
                input_path = glob.glob(mol2_dir + "/" + pdb + "*/" + file)[0]
                if input_path:
                    cmd.load(input_path, pdb + "_" + file)
                    cmd.show("cartoon", pdb + "_" + file)
                else:
                    print("Warning: File not found: " + file)

            # Show protein and site as cartoon
            cmd.show("cartoon", pdb + "_protein.mol")
            cmd.hide("lines", pdb + "_protein.mol2")
            cmd.show("cartoon", pdb + "_site.mol")
            cmd.hide("lines", pdb + "_site.mol2")

            # Show ligand as sticks
            cmd.show("sticks", pdb + "_ligand.mol2")
            cmd.hide("lines", pdb + "_ligand.mol2")

            # Highlight scPDB binding site
            res_list = color_scpdb_residues(pdb, mol2_dir, "site.mol2")
            cmd.select(pdb + "_scPDB", pdb + "_protein.mol2 and resi " + "+".join(res_list))
            cmd.color("blue", pdb + "_scPDB")

            # Hide site.mol2
            cmd.disable(pdb + "_site.mol2")

            # Load cgo files with reference points
            cgo_list = sorted(glob.glob(cgo_dir + "/" + pdb[:4] + "*"))
            print(cgo_list)
            if not cgo_list:
                print("Error: PDB ID is not available as cgo file.")
            else:
                print("Run cgo files...")
                for cgo in cgo_list:
                    cmd.run(cgo)
                    cmd.disable(cgo.split("/")[-1].replace(".py", ""))

            cmd.group(pdb, pdb + "*")

    cmd.zoom()


cmd.extend("load_scpdb_files", load_scpdb_files)


########################################################################################################################
# Auxiliary function
########################################################################################################################

def color_scpdb_residues(pdb, mol2_dir, input_file):
    """

    :param pdb:
    :param mol2_dir:
    :param input_file:
    :return:
    """

    # Set scPDB mol2 file path
    input_path = glob.glob(mol2_dir + "/" + pdb + "*/" + input_file)

    # Load mol2 file
    pmols = []

    # In case of multiple entries in one mol2 file, include iteration step
    for mol2 in split_multimol2(input_path[0]):
        pmol = PandasMol2().read_mol2_from_list(mol2_lines=mol2[1], mol2_code=mol2[0])
        pmols.append(pmol)
    pmol = pmols[0]

    # Get residue IDs from scPDB file
    res_list = [i[3:] for i in list(set(pmol.df["subst_name"]))]
    # Sort residue IDs
    res_list.sort(reverse=False)

    return res_list
