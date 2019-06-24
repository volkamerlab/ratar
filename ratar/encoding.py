"""
encoding.py

Read-across the targetome -
An integrated structure- and ligand-based workbench for computational target prediction and novel tool compound design

Handles the primary functions for encoding one or multiple binding site (s).
"""


import glob
import logging
from pathlib import Path
import pickle
import sys

from flatten_dict import flatten, unflatten
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import cbrt
from scipy.stats.stats import skew
import warnings

from ratar.auxiliary import MoleculeLoader, AminoAcidDescriptors
from ratar.auxiliary import create_directory, load_pseudocenters

warnings.simplefilter('error', FutureWarning)

logger = logging.getLogger(__name__)


class BindingSite:
    """
    Class used to represent a molecule and its encoding.

    Parameters
    ----------
    pmol : biopandas.mol2.pandas_mol2.PandasMol2 or biopandas.pdb.pandas_pdb.PandasPdb
        Content of mol2 or pdb file as BioPandas object.

    Attributes
    ----------
    molecule_id : str
        Molecule ID (e.g. PDB ID).
    molecule : DataFrame
        Data extracted from e.g. mol2 or pdb file.
    representatives : ratar.encoding.Representatives
        Representative atoms of binding site for different representation methods.
    shapes : Shapes
        Encoded binding site (reference points, distance distribution and distribution moments).

    Examples
    --------
    >>> from ratar.auxiliary import MoleculeLoader
    >>> from ratar.encoding import Representatives, Coordinates

    >>> molecule_path = '/path/to/pdb/or/mol2'

    >>> molecule_loader = MoleculeLoader()
    >>> molecule_loader.load_molecule(molecule_path, remove_solvent=True)
    >>> molecule = molecule_loader.get_first_molecule()

    >>> binding_site = BindingSite(molecule)
    """

    def __init__(self, pmol):

        self.molecule_id = pmol.code
        self.molecule = pmol.df
        self.representatives = self.get_representatives()
        self.shapes = self.run()

    def __eq__(self, other):
        """
        Check if two BindingSite objects are equal.
        """

        rules = [
            self.molecule_id == other.molecule_id,
            self.molecule.equals(other.molecule),
            self.representatives == other.representatives,
            self.shapes == other.shapes
        ]
        return all(rules)

    def get_representatives(self):
        representatives = Representatives(self.molecule_id)
        representatives.get_representatives(self.molecule)
        return representatives

    def get_coordinates(self, representatives):
        coordinates = Coordinates(self.molecule_id)
        coordinates.get_coordinates(representatives)
        return coordinates

    def get_physicochemicalproperties(self, representatives):
        physicochemicalproperties = PhysicoChemicalProperties(self.molecule_id)
        physicochemicalproperties.get_physicochemicalproperties(representatives)
        return physicochemicalproperties

    def get_subsets(self, representatives):
        subsets = Subsets(self.molecule_id)
        subsets.get_pseudocenter_subsets_indices(representatives)
        return subsets

    def get_points(self, coordinates, physicochemicalproperties, subsets):
        points = Points(self.molecule_id)
        points.get_points(coordinates, physicochemicalproperties)
        points.get_points_pseudocenter_subsets(subsets)
        return points

    def get_shapes(self, points):
        shapes = Shapes(self.molecule_id)
        shapes.get_shapes(points)
        shapes.get_shapes_pseudocenter_subsets(points)
        return shapes

    def run(self):
        representatives = self.get_representatives()
        coordinates = self.get_coordinates(representatives)
        physicochemicalproperties = self.get_physicochemicalproperties(representatives)
        subsets = self.get_subsets(representatives)
        points = self.get_points(coordinates, physicochemicalproperties, subsets)
        shapes = self.get_shapes(points)
        return shapes


class Representatives:
    """
    Class used to store molecule representatives. Representatives are selected atoms in a binding site,
    e.g. all Calpha atoms of a binding site could serve as its representatives.

    Attributes
    ----------
    molecule_id : str
        Molecule ID (e.g. PDB ID).
    data : dict of pandas.DataFrames
        Dictionary (representatives types, e.g. 'pc') of DataFrames containing molecule structural data.

    Examples
    --------
    >>> from ratar.auxiliary import MoleculeLoader
    >>> from ratar.encoding import Representatives

    >>> molecule_path = '/path/to/pdb/or/mol2'

    >>> molecule_loader = MoleculeLoader()
    >>> molecule_loader.load_molecule(molecule_path, remove_solvent=True)
    >>> pmol = molecule_loader.get_first_molecule()

    >>> representatives = Representatives()
    >>> representatives.get_representatives_from_pmol(pmol)
    >>> representatives
    """

    def __init__(self, molecule_id=None):

        self.molecule_id = molecule_id
        self.data = {
            'ca':  pd.DataFrame(),
            'pca':  pd.DataFrame(),
            'pc':  pd.DataFrame(),
        }

    @property
    def ca(self):
        return self.data['ca']

    @property
    def pca(self):
        return self.data['pca']

    @property
    def pc(self):
        return self.data['pc']

    def __eq__(self, other):
        """
        Check if two Representatives objects are equal.
        Return True if dictionary keys (strings) and dictionary values (DataFrames) are equal, else return False.
        """

        obj1 = flatten(self.data, reducer='path')
        obj2 = flatten(other.data, reducer='path')

        try:
            rules = [
                obj1.keys() == obj2.keys(),
                all([v.equals(obj2[k]) for k, v in obj1.items()])
            ]
        except KeyError:
            rules = False

        return all(rules)

    def get_representatives_from_pmol(self, pmol):
        """
        Convenience class method: Get representatives from pmol object.

        Parameters
        ----------
        pmol : biopandas.mol2.pandas_mol2.PandasMol2 or biopandas.pdb.pandas_pdb.PandasPdb
        Content of mol2 or pdb file as BioPandas object.
        """

        self.get_representatives(pmol.df)

    def get_representatives(self, molecule):
        """
        Extract binding site representatives.

        Parameters
        ----------
        molecule : pandas.DataFrame
            DataFrame containing atom lines from input file.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing atom lines from input file described by Z-scales.
        """

        for key in self.data.keys():
            if key == 'ca':
                self.data['ca'] = self._get_ca(molecule)
            elif key == 'pca':
                self.data['pca'] = self._get_pca(molecule)
            elif key == 'pc':
                self.data['pc'] = self._get_pc(molecule)
            else:
                raise KeyError(f'Unknown representative key: {key}')

        return self.data

    @staticmethod
    def _get_ca(mol):
        """
        Extract Calpha atoms from binding site.

        Parameters
        ----------
        mol : pandas.DataFrame
            DataFrame containing atom lines from input file.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing atom lines from input file described by Z-scales.
        """

        # Filter for atom name 'CA' (first condition) but exclude calcium (second condition)
        bs_ca = mol[(mol['atom_name'] == 'CA') & (mol['res_name'] != 'CA')]

        return bs_ca

    @staticmethod
    def _get_pca(mol):
        """
        Extract pseudocenter atoms from molecule.

        Parameters
        ----------
        mol: pandas.DataFrame
            DataFrame containing atom lines from input file.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing atom lines from input file that belong to pseudocenters.
        """

        # Load pseudocenter atoms
        pseudocenter_atoms = load_pseudocenters()

        # Collect all molecule atoms that belong to pseudocenters
        mol_pca_subset = []

        for index, row in mol.iterrows():

            query = f'{row["res_name"]}_{row["atom_name"]}'

            if query in list(pseudocenter_atoms['pc_atom_pattern']):  # Non-peptide bond atoms

                pc_ix = pseudocenter_atoms.index[pseudocenter_atoms['pc_atom_pattern'] == query].tolist()[0]

                mol_pca_subset.append(
                    [
                        index,
                        pseudocenter_atoms.iloc[pc_ix]['pc_type'],
                        pseudocenter_atoms.iloc[pc_ix]['pc_id'],
                        pseudocenter_atoms.iloc[pc_ix]['pc_atom_id']
                    ]
                )

            elif row['atom_name'] == 'O':  # Peptide bond atoms

                mol_pca_subset.append(
                    [
                        index,
                        'HBA',
                        'PEP_HBA_1',
                        'PEP_HBA_1_0'
                    ]
                )

            elif row['atom_name'] == 'N':  # Peptide bond atoms

                mol_pca_subset.append(
                    [
                        index,
                        'HBD',
                        'PEP_HBD_1',
                        'PEP_HBD_1_N']
                )

            elif row['atom_name'] == 'C':  # Peptide bond atoms

                mol_pca_subset.append(
                    [
                        index,
                        'AR',
                        'PEP_AR_1',
                        'PEP_AR_1_C'
                    ]
                )

        # Cast list of lists to DataFrame
        mol_pca_subset_df = pd.DataFrame(mol_pca_subset)
        mol_pca_subset_df.columns = ['index', 'pc_type', 'pc_id', 'pc_atom_id']
        mol_pca_subset_df.index = [i[0] for i in mol_pca_subset]
        mol_pca_subset_df.drop(columns=['index'], inplace=True)

        # Join molecule with pseudocenter subset
        mol_pca_subset_df = mol.join(mol_pca_subset_df, how='outer')
        mol_pca_subset_df.dropna(how='any', inplace=True)

        return mol_pca_subset_df

    def _get_pc(self, mol):
        """
        Extract pseudocenters from molecule.

        Parameters
        ----------
        mol : pandas.DataFrame
            DataFrame containing atom lines from input file.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing atom lines for (aggregated) pseudocenters, i.e. aggregate multiple atoms belonging to
            one pseudocenter.
        """

        # Get pseudocenter atoms
        mol_pca_subset_df = self._get_pca(mol)

        # Calculate pseudocenters
        mol_pc_subset = []

        for key, group in mol_pca_subset_df.groupby(['subst_name', 'pc_id'], sort=False):

            if len(group) == 1:  # If pseudocenter only contains one atom, keep data
                row = group.iloc[0].copy()
                row['atom_id'] = [row['atom_id']]
                row['atom_name'] = [row['atom_name']]

                mol_pc_subset.append(row)

            else:  # If pseudocenter contains multiple atoms, aggregate data
                first_row = group.iloc[0].copy()
                first_row['atom_id'] = list(group['atom_id'])
                first_row['atom_name'] = list(group['atom_name'])
                first_row['x'] = group['x'].mean()
                first_row['y'] = group['y'].mean()
                first_row['z'] = group['z'].mean()
                first_row['charge'] = group['charge'].mean()  # TODO Alternatives to mean of a charge?

                mol_pc_subset.append(first_row)

        mol_pc_subset_df = pd.concat(mol_pc_subset, axis=1).T

        return mol_pc_subset_df


class Coordinates:
    """
    Class used to store the coordinates of molecule representatives, which were defined by the Representatives class.

    Attributes
    ----------
    molecule_id : str
        Molecule ID (e.g. PDB ID).
    data : dict of pandas.DataFrames
        Dictionary (representatives types, e.g. 'pc') of DataFrames containing coordinates.

    Examples
    --------
    >>> from ratar.auxiliary import MoleculeLoader
    >>> from ratar.encoding import Representatives, Coordinates

    >>> molecule_path = '/path/to/pdb/or/mol2'

    >>> molecule_loader = MoleculeLoader()
    >>> molecule_loader.load_molecule(molecule_path, remove_solvent=True)
    >>> pmol = molecule_loader.get_first_molecule()

    >>> coordinates = Coordinates()
    >>> coordinates.get_coordinates_from_pmol(pmol)
    """

    def __init__(self, molecule_id=None):

        self.molecule_id = molecule_id
        self.data = {
            'ca': None,
            'pca': None,
            'pc': None,
        }

    @property
    def ca(self):
        return self.data['ca']

    @property
    def pca(self):
        return self.data['pca']

    @property
    def pc(self):
        return self.data['pc']

    def __eq__(self, other):
        """
        Check if two Coordinates objects are equal.
        """

        obj1 = flatten(self.data, reducer='path')
        obj2 = flatten(other.coord_dict, reducer='path')

        try:
            rules = [
                obj1.keys() == obj2.keys(),
                all([v.equals(obj2[k]) for k, v in obj1.items()])
            ]
        except KeyError:
            rules = False

        return all(rules)

    def get_coordinates_from_pmol(self, pmol):
        """
        Convenience class method: Get coordinates from pmol object.

        Parameters
        ----------
        pmol : biopandas.mol2.pandas_mol2.PandasMol2 or biopandas.pdb.pandas_pdb.PandasPdb
        Content of mol2 or pdb file as BioPandas object.
        """

        representatives = Representatives()
        representatives.get_representatives(pmol.df)

        self.get_coordinates(representatives)

    def get_coordinates(self, representatives):
        """
        Get coordinates (x, y, z) for molecule reprentatives.

        Parameters
        ----------
        representatives : ratar.encoding.Representatives
            Representatives class instance.

        Returns
        -------
        dict of DataFrames
            Dictionary (representatives types, e.g. 'pc') of DataFrames containing molecule coordinates.
        """

        self.data = {}

        for k1, v1 in representatives.data.items():
            if isinstance(v1, pd.DataFrame):
                self.data[k1] = v1[['x', 'y', 'z']]
            elif isinstance(v1, dict):
                self.data[k1] = {k2: v2[['x', 'y', 'z']] for (k2, v2) in v1.items()}
            else:
                raise TypeError(f'Expected dict or pandas.DataFrame but got {type(v1)}')

        return self.data


class PhysicoChemicalProperties:
    """
    Class used to store the physicochemical properties of molecule representatives, which were defined by the
    Representatives class.

    Attributes
    ----------
    molecule_id : str
        Molecule ID (e.g. PDB ID).
    data : dict of dict of pandas.DataFrame
        Dictionary (representatives types, e.g. 'pc') of dictionaries (physicochemical properties types, e.g. 'z123') of
        DataFrames containing physicochemical properties.

    Examples
    --------
    >>> from ratar.auxiliary import MoleculeLoader
    >>> from ratar.encoding import Representatives, Coordinates

    >>> molecule_path = '/path/to/pdb/or/mol2'

    >>> molecule_loader = MoleculeLoader()
    >>> molecule_loader.load_molecule(molecule_path, remove_solvent=True)
    >>> pmol = molecule_loader.get_first_molecule()

    >>> physicochemicalproperties = PhysicoChemicalProperties()
    >>> physicochemicalproperties.get_physicochemicalproperties_from_pmol(pmol)
    """

    def __init__(self, molecule_id=None):

        self.molecule_id = molecule_id
        self.data = {
            'ca': {},
            'pca': {},
            'pc': {},
        }

    @property
    def ca(self):
        return self.data['ca']

    @property
    def pca(self):
        return self.data['pca']

    @property
    def pc(self):
        return self.data['pc']

    def __eq__(self, other):
        """
        Check if two PhysicoChemicalProperties objects are equal.
        """

        obj1 = flatten(self.data, reducer='path')
        obj2 = flatten(other.data, reducer='path')

        try:
            rules = [
                obj1.keys() == obj2.keys(),
                all([v.equals(obj2[k]) for k, v in obj1.items()])
            ]
        except KeyError:
            rules = False

        return all(rules)

    def get_physicochemicalproperties_from_pmol(self, pmol):
        """
        Convenience class method: Get physicochemical properties from pmol object.

        Parameters
        ----------
        pmol : biopandas.mol2.pandas_mol2.PandasMol2 or biopandas.pdb.pandas_pdb.PandasPdb
        Content of mol2 or pdb file as BioPandas object.
        """

        representatives = Representatives()
        representatives.get_representatives(pmol.df)

        self.get_physicochemicalproperties(representatives)

    def get_physicochemicalproperties(self, representatives):
        """
        Extract physicochemical properties (main function).

        Parameters
        ----------
        representatives : ratar.encoding.Representatives
            Representatives class instance.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing physicochemical properties.
        """

        self.data = {}

        physicochemicalproperties_keys = ['z1', 'z123']

        for k1, v1 in representatives.data.items():
            self.data[k1] = {}

            for k2 in physicochemicalproperties_keys:
                if k2 == 'z1':
                    self.data[k1][k2] = self._get_zscales(v1, 1)
                elif k2 == 'z123':
                    self.data[k1][k2] = self._get_zscales(v1, 3)
                else:
                    raise KeyError(f'Unknown representatives key: {k2}. '
                                   f'Select: {", ".join(physicochemicalproperties_keys)}')

        return self.data

    def _get_zscales(self, representatives_df, z_number):
        """
        Extract Z-scales from binding site representatives.

        Parameters
        ----------
        representatives_df : pandas.DataFrame
            Representatives' data for a certain representatives type.
        z_number : int
            Number of Z-scales to be included.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing Z-scales.
        """

        # Load amino acid to Z-scales transformation matrix
        aminoacid_descriptors = AminoAcidDescriptors()
        zscales = aminoacid_descriptors.zscales

        # Get Z-scales for representatives' amino acids
        representatives_zscales = []
        for index, row in representatives_df.iterrows():
            aminoacid = row['res_name']

            if aminoacid in zscales.index:
                representatives_zscales.append(zscales.loc[aminoacid])
            else:
                representatives_zscales.append(pd.Series([None]*zscales.shape[1], index=zscales.columns))

                # Log missing Z-scales
                logger.info(f'The following atom (residue) has no Z-scales assigned: {row["subst_name"]}',
                            extra={'molecule_id': self.molecule_id})

        # Cast into DataFrame
        representatives_zscales_df = pd.DataFrame(
            representatives_zscales,
            index=representatives_df.index,
            columns=zscales.columns
        )

        return representatives_zscales_df.iloc[:, :z_number]


class Subsets:
    """
    Class used to store subset indices (DataFrame indices) of molecule representatives, which were defined by the
    Representatives class.

    Attributes
    ----------
    molecule_id : str
        Molecule ID (e.g. PDB ID).
    data_pseudocenter_subsets : dict of dict of list
        Dictionary (representatives types, e.g. 'pc') of dictionaries (pseudocenter subset types, e.g. 'HBA') of
        lists containing subset indices.

    Examples
    --------
    >>> from ratar.auxiliary import MoleculeLoader
    >>> from ratar.encoding import Representatives, Coordinates

    >>> molecule_path = '/path/to/pdb/or/mol2'

    >>> molecule_loader = MoleculeLoader()
    >>> molecule_loader.load_molecule(molecule_path, remove_solvent=True)
    >>> pmol = molecule_loader.get_first_molecule()

    >>> subsets = Subsets()
    >>> subsets.get_pseudocenter_subsets_indices_from_pmol(pmol)
    """

    def __init__(self, molecule_id=None):

        self.molecule_id = molecule_id
        self.data_pseudocenter_subsets = {
            'pca': {},
            'pc': {}
        }

    @property
    def pseudocenters(self):
        return self.data_pseudocenter_subsets['pc']

    @property
    def pseudocenter_atoms(self):
        return self.data_pseudocenter_subsets['pca']

    def __eq__(self, other):
        """
        Check if two Subsets objects are equal.
        """

        obj1 = flatten(self.data_pseudocenter_subsets, reducer='path')
        obj2 = flatten(other.data_pseudocenter_subsets, reducer='path')

        try:
            rules = [
                obj1.keys() == obj2.keys(),
                all([v.equals(obj2[k]) for k, v in obj1.items()])
            ]
        except KeyError:
            rules = False

        return all(rules)

    def get_pseudocenter_subsets_indices_from_pmol(self, pmol):
        """
        Convenience class method: Get pseudocenter subsets indices from pmol object.

        Parameters
        ----------
        pmol : biopandas.mol2.pandas_mol2.PandasMol2 or biopandas.pdb.pandas_pdb.PandasPdb
        Content of mol2 or pdb file as BioPandas object.
        """

        representatives = Representatives()
        representatives.get_representatives(pmol.df)

        self.get_pseudocenter_subsets_indices(representatives)

    def get_pseudocenter_subsets_indices(self, representatives):
        """
        Extract feature subsets from pseudocenters (pseudocenter atoms).

        Parameters
        ----------
        representatives : ratar.encoding.Representatives
            Representatives class instance.

        Returns
        -------
        dict of dict of list of int
            List of DataFrame indices in a dictionary (representatives types, e.g. 'pc') of dictionaries (pseudocenter
            subset types, e.g. 'HBA').
        """

        # Load pseudocenter atoms
        pseudocenter_atoms = load_pseudocenters(remove_hbda=True)

        self.data_pseudocenter_subsets = {}

        for k1 in ['pc', 'pca']:

            self.data_pseudocenter_subsets[k1] = {}

            repres = representatives.data[k1]

            # Loop over all pseudocenter subset types
            for k2 in list(set(pseudocenter_atoms['pc_type'])):

                # If pseudocenter type exists in dataset, save corresponding subset, else save None
                if k2 in set(repres['pc_type']):
                    self.data_pseudocenter_subsets[k1][k2] = list(repres[repres['pc_type'] == k2].index)
                else:
                    self.data_pseudocenter_subsets[k1][k2] = []

        return self.data_pseudocenter_subsets


class Points:
    """
    Class used to store the coordinates and (optionally) physicochemical properties of molecule representatives, which
    were defined by the Representatives class.

    Binding site representatives (i.e. atoms) can have different dimensions, for instance an atom can have
    - 3 dimensions (spatial properties x, y, z) or
    - more dimensions (spatial and some additional properties).

    Attributes
    ----------
    molecule_id : str
        Molecule ID (e.g. PDB ID).
    data : dict of dict of pandas.DataFrames
         Dictionary (representatives types, e.g. 'pc') of dictionaries (physicochemical properties types, e.g. 'z1')
         of DataFrames containing coordinates and physicochemical properties.
    data_pseudocenter_subsets : dict of dict of dict of pandas.DataFrames
        Dictionary (representatives types, e.g. 'pc') of dictionaries (physicochemical properties, e.g. 'pc_z123')
        of dictionaries (subset types, e.g. 'HBA') containing each a DataFrame describing the subsetted atoms.

    Examples
    --------
    >>> from ratar.auxiliary import MoleculeLoader
    >>> from ratar.encoding import Representatives, Coordinates

    >>> molecule_path = '/path/to/pdb/or/mol2'

    >>> molecule_loader = MoleculeLoader()
    >>> molecule_loader.load_molecule(molecule_path, remove_solvent=True)
    >>> pmol = molecule_loader.get_first_molecule()

    >>> points = Points()
    >>> points.get_points_from_pmol(pmol)
    """

    def __init__(self, molecule_id=None):

        self.molecule_id = molecule_id
        self.data = {
            'ca': {},
            'pca': {},
            'pc': {}
        }
        self.data_pseudocenter_subsets = {
            'pc': {},
            'pca': {}
        }

    @property
    def ca(self):
        return self.data['ca']

    @property
    def pca(self):
        return self.data['pca']

    @property
    def pc(self):
        return self.data['pc']

    @property
    def pca_subsets(self):
        return self.data_pseudocenter_subsets['pca']

    @property
    def pc_subsets(self):
        return self.data_pseudocenter_subsets['pc']

    def __eq__(self, other):
        """
        Check if two Points objects are equal.
        """

        obj1 = (
            flatten(self.data, reducer='path'),
            flatten(self.data_pseudocenter_subsets, reducer='path')
        )

        obj2 = (
            flatten(other.data, reducer='path'),
            flatten(other.data_pseudocenter_subsets, reducer='path')
        )

        try:
            rules = [
                obj1[0].keys() == obj2[0].keys(),
                obj1[1].keys() == obj2[1].keys(),
                all([v.equals(obj2[0][k]) for k, v in obj1[0].items()]),
                all([v.equals(obj2[1][k]) for k, v in obj1[1].items()])
            ]
        except KeyError:
            rules = False

        return all(rules)

    def get_points_from_pmol(self, pmol):
        """
        Convenience class method: Get points from pmol object.

        Parameters
        ----------
        pmol : biopandas.mol2.pandas_mol2.PandasMol2 or biopandas.pdb.pandas_pdb.PandasPdb
        Content of mol2 or pdb file as BioPandas object.
        """

        representatives = Representatives()
        representatives.get_representatives(pmol.df)

        coordinates = Coordinates()
        coordinates.get_coordinates(representatives)

        physicochemicalproperties = PhysicoChemicalProperties()
        physicochemicalproperties.get_physicochemicalproperties(representatives)

        subsets = Subsets()
        subsets.get_pseudocenter_subsets_indices(representatives)

        self.get_points(coordinates, physicochemicalproperties)
        self.get_points_pseudocenter_subsets(subsets)

    def get_points(self, coordinates, physicochemicalproperties):
        """
        Concatenate spatial (3-dimensional) and physicochemical (N-dimensional) properties
        to an 3+N-dimensional vector for each point in dataset (i.e. representative atoms in a binding site).

        Parameters
        ----------
        coordinates : ratar.encoding.Coordinates
            Coordinates class instance.
        physicochemicalproperties : ratar.PhysicoChemicalProperties
            PhysicoChemicalProperties class instance.

        Returns
        -------
        dict of dict of pandas.DataFrames
             Dictionary (representatives types, e.g. 'pc') of dictionaries (physicochemical properties types, e.g. 'z1')
             of DataFrames containing coordinates and physicochemical properties.
        """

        self.data = {}

        # Get physicochemical properties
        physicochemicalproperties_keys = physicochemicalproperties.data['ca'].keys()

        for k1 in coordinates.data.keys():

            self.data[k1] = {}

            # Add points without physicochemical properties
            self.data[k1]['no'] = coordinates.data[k1].copy()

            # Drop rows (atoms) with empty entries (e.g. atoms without Z-scales assignment)
            self.data[k1]['no'].dropna(inplace=True)

            for k2 in physicochemicalproperties_keys:
                self.data[k1][k2] = pd.concat(
                    [
                        coordinates.data[k1],
                        physicochemicalproperties.data[k1][k2]
                    ],
                    axis=1
                )

                # Drop rows (atoms) with empty entries (e.g. atoms without Z-scales assignment)
                self.data[k1][k2].dropna(inplace=True)

        return self.data

    def get_points_pseudocenter_subsets(self, subsets):
        """
        Get

        Parameters
        ----------
        subsets : ratar.encoding.Subsets
            Subsets class instance.

        Returns
        -------
        dict of dict of dict of pandas.DataFrames
            Dictionary (representatives types, e.g. 'pc') of dictionaries (physicochemical properties, e.g. 'pc_z123')
            of dictionaries (subset types, e.g. 'HBA') containing each a DataFrame describing the subsetted atoms.
        """

        self.data_pseudocenter_subsets = {}

        for k1, v1 in self.data.items():  # Representatives

            # Select points keys that we want to subset, e.g. we want to subset pseudocenters but not Calpha atoms
            if k1 in subsets.data_pseudocenter_subsets.keys():
                self.data_pseudocenter_subsets[k1] = {}

                for k2, v2 in v1.items():  # Physicochemical properties
                    self.data_pseudocenter_subsets[k1][k2] = {}

                    for k3, v3 in subsets.data_pseudocenter_subsets[k1].items():
                        # Select only subsets indices that are in points
                        # In points, e.g.amino acid atoms with missing Z-scales are discarded
                        labels = v2.index.intersection(v3)
                        self.data_pseudocenter_subsets[k1][k2][k3] = v2.loc[labels, :]

        return self.data_pseudocenter_subsets


class Shapes:
    """
    Class used to store the encoded molecule representatives, which were defined by the Representatives class.

    Attributes
    ----------
    molecule_id : str
        Molecule ID (e.g. PDB ID).
    data : dict of dict of dict of dict of pandas.DataFrames
        Dictionary (representatives types, e.g. 'pc') of dictionaries (physicochemical properties types, e.g. 'z1')
        of dictionaries (encoding method, e.g. '3Dusr') of dictionaries containing DataFrames for the encoding:
        'ref_points' (the reference points), 'distances' (the distances from reference points to representatives),
        and 'moments' (the first three moments for the distance distribution).
    data_pseudocenter_subsets : dict of dict of dict of dict of pandas.DataFrames
        Dictionary (representatives types, e.g. 'pc') of dictionaries (physicochemical properties types, e.g. 'z1')
        of dictionaries (encoding method, e.g. '3Dusr') of dictionaries (subsets types, e.g. 'HBA') of dictionaries
        containing DataFrames for the encoding: 'ref_points' (the reference points), 'distances' (the distances from
        reference points to representatives), and 'moments' (the first three moments for the distance distribution).

    Examples
    --------
    >>> from ratar.auxiliary import MoleculeLoader
    >>> from ratar.encoding import Representatives, Coordinates

    >>> molecule_path = '/path/to/pdb/or/mol2'

    >>> molecule_loader = MoleculeLoader()
    >>> molecule_loader.load_molecule(molecule_path, remove_solvent=True)
    >>> pmol = molecule_loader.get_first_molecule()

    >>> shapes = Shapes()
    >>> shapes.get_shapes_from_pmol(pmol)
    """

    def __init__(self, molecule_id=None):

        self.molecule_id = molecule_id
        self.data = {
            'ca': {},
            'pca': {},
            'pc': {}
        }
        self.data_pseudocenter_subsets = {
            'pc': {},
            'pca': {}
        }

    @property
    def ca(self):
        return self.data['ca']

    @property
    def pca(self):
        return self.data['pca']

    @property
    def pc(self):
        return self.data['pc']

    @property
    def pca_subsets(self):
        return self.data_pseudocenter_subsets['pca']

    @property
    def pc_subsets(self):
        return self.data_pseudocenter_subsets['pc']

    def __eq__(self, other):
        """
        Check if two Shapes objects are equal.
        """

        obj1 = (
            flatten(self.data, reducer='path'),
            flatten(self.data_pseudocenter_subsets, reducer='path')
        )

        obj2 = (
            flatten(other.data, reducer='path'),
            flatten(other.data_pseudocenter_subsets, reducer='path')
        )

        try:
            rules = [
                obj1[0].keys() == obj2[0].keys(),
                obj1[1].keys() == obj2[1].keys(),
                all([v.equals(obj2[0][k]) for k, v in obj1[0].items()]),
                all([v.equals(obj2[1][k]) for k, v in obj1[1].items()])
            ]
        except KeyError:
            rules = False

        return all(rules)

    def get_shapes_from_pmol(self, pmol):
        """
        Convenience class method: Get shapes from pmol object.

        Parameters
        ----------
        pmol : biopandas.mol2.pandas_mol2.PandasMol2 or biopandas.pdb.pandas_pdb.PandasPdb
        Content of mol2 or pdb file as BioPandas object.
        """

        representatives = Representatives()
        representatives.get_representatives(pmol)

        coordinates = Coordinates()
        coordinates.get_coordinates(representatives)

        physicochemicalproperties = PhysicoChemicalProperties()
        physicochemicalproperties.get_physicochemicalproperties(representatives)

        subsets = Subsets()
        subsets.get_pseudocenter_subsets_indices(representatives)

        points = Points()
        points.get_points(coordinates, physicochemicalproperties)
        points.get_points_pseudocenter_subsets(subsets)

        self.get_shapes(points)

    def get_shapes(self, points):
        """
        Get the encoding of a molecule for different types of representatives, physicochemical properties, and encoding
        methods.

        Parameters
        ----------
        points : ratar.encoding.Points
            Points class instance

        Returns
        -------
        dict of dict of dict of dict of pandas.DataFrames
            Dictionary (representatives types, e.g. 'pc') of dictionaries (physicochemical properties types, e.g. 'z1')
            of dictionaries (encoding method, e.g. '3Dusr') of dictionaries containing DataFrames for the encoding:
            'ref_points' (the reference points), 'distances' (the distances from reference points to representatives),
            and 'moments' (the first three moments for the distance distribution).
        """

        self.data = {}

        # Flatten nested dictionary
        points_flat = flatten(points.data, reducer='path')

        for k, v in points_flat.items():
            self.data[k] = self._get_shape_by_method(v)

        # Unflatten dictionary back to nested dictionary
        self.data = unflatten(self.data, splitter='path')

        return self.data

    def get_shapes_pseudocenter_subsets(self, points):
        """

        Parameters
        ----------
        points : ratar.encoding.Points
            Points class instance.

        Returns
        -------
        dict of dict of dict of dict of pandas.DataFrames
            Dictionary (representatives types, e.g. 'pc') of dictionaries (physicochemical properties types, e.g. 'z1')
            of dictionaries (encoding method, e.g. '3Dusr') of dictionaries (subsets types, e.g. 'HBA') of dictionaries
            containing DataFrames for the encoding: 'ref_points' (the reference points), 'distances' (the distances from
            reference points to representatives), and 'moments' (the first three moments for the distance distribution).

        """

        self.data_pseudocenter_subsets = {}

        # Flatten nested dictionary
        points_flat = flatten(points.data_pseudocenter_subsets, reducer='path')

        for k, v in points_flat.items():
            self.data_pseudocenter_subsets[k] = self._get_shape_by_method(v)

        # Change key order of (flattened) nested dictionary (reverse subset type and encoding type)
        self.data_pseudocenter_subsets = self._reorder_subset_keys()

        # Unflatten dictionary back to nested dictionary
        self.data_pseudocenter_subsets = unflatten(self.data_pseudocenter_subsets, splitter='path')

        return self.data_pseudocenter_subsets

    def _get_shape_by_method(self, points_df):
        """
        Apply encoding method on points depending on points dimensions and return encoding.

        Parameters
        ----------
        points_df : pandas.DataFrame
            DataFrame containing points which can have different dimensions.

        Returns
        -------
        dict of dict of pandas.DataFrames
            Dictionary (encoding type) of dictionaries containing DataFrames for the encoding:
            'ref_points' (the reference points), 'distances' (the distances from reference points to representatives),
            and 'moments' (the first three moments for the distance distribution).
        """

        n_points = points_df.shape[0]
        n_dimensions = points_df.shape[1]

        # Select method based on number of dimensions and points
        if n_dimensions == 3 and n_points > 3:
            return {'3Dusr': self._calc_shape_3dim_usr(points_df),
                    '3Dcsr': self._calc_shape_3dim_csr(points_df)}
        elif n_dimensions == 4 and n_points > 4:
            return {'4Delectroshape': self._calc_shape_4dim_electroshape(points_df)}
        elif n_dimensions == 6 and n_points > 6:
            return {'6Dratar1': self._calc_shape_6dim(points_df)}
        elif n_dimensions < 3:
            logger.error(f'Unexpected points dimension: {points_df.shape[1]}. Not implemented.',
                         extra={'molecule_id': self.molecule_id})
            return None
        elif n_dimensions == 3 and n_points <= 3:
            logger.error(f'Number of points in 3D must be at least 4. Number of input points: {points_df.shape[0]}.',
                         extra={'molecule_id': self.molecule_id})
            return {'3Dusr': {'ref_points': None, 'dist': None, 'moments': None},
                    '3Dcsr': {'ref_points': None, 'dist': None, 'moments': None}}
        elif n_dimensions == 4 and n_points <= 4:
            logger.error(f'Number of points in 4D must be at least 5. Number of input points: {points_df.shape[0]}.',
                         extra={'molecule_id': self.molecule_id})
            return {'4Delectroshape': {'ref_points': None, 'dist': None, 'moments': None}}
        elif n_dimensions == 6 and n_points <= 6:
            logger.error(f'Number of points in 6D must be at least 7. Number of input points: {points_df.shape[0]}.',
                         extra={'molecule_id': self.molecule_id})
            return {'6Dratar1': {'ref_points': None, 'dist': None, 'moments': None}}
        elif n_dimensions > 7:
            logger.error(f'Unexpected points dimension: {points_df.shape[1]}. Not implemented.',
                         extra={'molecule_id': self.molecule_id})
            return None

    def _calc_shape_3dim_usr(self, points):
        """
        Encode binding site (3-dimensional points) based on the USR method.

        Parameters
        ----------
        points : pandas.DataFrame
            Coordinates of representatives (points) in binding site.

        Returns
        -------
        dict
            Reference points, distance distributions, and moments.

        Notes
        -----
        1. Calculate reference points for a set of points:
           - c1, centroid
           - c2, closest atom to c1
           - c3, farthest atom to c1
           - c4, farthest atom to c3
        2. Calculate distances (distance distribution) from reference points to all other points.
        3. Calculate first, second, and third moment for each distance distribution.

        References
        ----------
        [1]_ Ballester and Richards, "Ultrafast shape Recognition to search compound databases for similar molecular
        shapes", J Comput Chem, 2007.
        """

        if points.shape[1] != 3:
            sys.exit(f'Error: Dimension of input (points) is not {points.shape[1]} as requested.')

        # Get centroid of input coordinates
        c1 = points.mean(axis=0)

        # Get distances from c1 to all other points
        dist_c1 = self._calc_distances_to_point(points, c1)

        # Get closest and farthest atom to centroid c1
        c2, c3 = points.loc[dist_c1.idxmin()], points.loc[dist_c1.idxmax()]

        # Get distances from c2 to all other points, get distances from c3 to all other points
        dist_c2 = self._calc_distances_to_point(points, c2)
        dist_c3 = self._calc_distances_to_point(points, c3)

        # Get farthest atom to farthest atom to centroid c3
        c4 = points.loc[dist_c3.idxmax()]

        # Get distances from c4 to all other points
        dist_c4 = self._calc_distances_to_point(points, c4)

        c = [c1, c2, c3, c4]
        dist = [dist_c1, dist_c2, dist_c3, dist_c4]

        return self._get_shape_dict(c, dist)

    def _calc_shape_3dim_csr(self, points):
        """
        Encode binding site (3-dimensional points) based on the CSR method.

        Parameters
        ----------
        points : pandas.DataFrame
            Coordinates of representatives (points) in binding site.

        Returns
        -------
        dict
            Reference points, distance distributions, and moments.

        Notes
        -----
        1. Calculate reference points for a set of points:
           - c1, centroid
           - c2, farthest atom to c1
           - c3, farthest atom to c2
           - c4, cross product of two vectors spanning c1, c2, and c3
        2. Calculate distances (distance distribution) from reference points to all other points.
        3. Calculate first, second, and third moment for each distance distribution.

        References
        ----------
        [1]_ Armstrong et al., "Molecular similarity including chirality", J Mol Graph Mod, 2009
        """

        if points.shape[1] != 3:
            sys.exit(f'Error: Dimension of input (points) is not {points.shape[1]} as requested.')

        # Get centroid of input coordinates, and distances from c1 to all other points
        c1 = points.mean(axis=0)
        dist_c1 = self._calc_distances_to_point(points, c1)

        # Get farthest atom to centroid c1, and distances from c2 to all other points
        c2 = points.loc[dist_c1.idxmax()]
        dist_c2 = self._calc_distances_to_point(points, c2)

        # Get farthest atom to farthest atom to centroid c2, and distances from c3 to all other points
        c3 = points.loc[dist_c2.idxmax()]
        dist_c3 = self._calc_distances_to_point(points, c3)

        # Get forth reference point, including chirality information
        # Define two vectors spanning c1, c2, and c3 (from c1)
        a = c2 - c1
        b = c3 - c1

        # Calculate cross product of a and b (keep same units and normalise vector to have half the norm of vector a)
        cross = np.cross(a, b)  # Cross product
        cross_norm = np.sqrt(sum(cross ** 2))  # Norm of cross product
        cross_unit = cross / cross_norm  # Cross unit vector
        a_norm = np.sqrt(sum(a ** 2))  # Norm of vector a
        c4 = pd.Series(a_norm / 2 * cross_unit, index=['x', 'y', 'z'])

        # Get distances from c4 to all other points
        dist_c4 = self._calc_distances_to_point(points, c4)

        c = [c1, c2, c3, c4]
        dist = [dist_c1, dist_c2, dist_c3, dist_c4]

        return self._get_shape_dict(c, dist)

    def _calc_shape_4dim_electroshape(self, points, scaling_factor=1):
        """
        Encode binding site (4-dimensional points) based on the ElectroShape method.

        Parameters
        ----------
        points : pandas.DataFrame
            Coordinates of representatives (points) in binding site.
        scaling_factor : int or float
            Scaling factor that can put higher or lower weight on non-spatial dimensions.

        Returns
        -------
        dict
            Reference points, distance distributions, and moments.

        Notes
        -----
        1. Calculate reference points for a set of points:
           - c1, centroid
           - c2, farthest atom to c1
           - c3, farthest atom to c2
           - c4 and c5, cross product of 2 vectors spanning c1, c2, and c3 with positive/negative sign in 4th dimension
        2. Calculate distances (distance distribution) from reference points to all other points.
        3. Calculate first, second, and third moment for each distance distribution.

        References
        ----------
        [1]_ Armstrong et al., "ElectroShape: fast molecular similarity calculations incorporating shape, chirality and
        electrostatics", J Comput Aided Mol Des, 2010.
        """

        if points.shape[1] != 4:
            sys.exit(f'Error: Dimension of input (points) is not {points.shape[1]} as requested.')

        # Get centroid of input coordinates (in 4 dimensions), and distances from c1 to all other points
        c1 = points.mean(axis=0)
        dist_c1 = self._calc_distances_to_point(points, c1)

        # Get farthest atom to centroid c1 (in 4 dimensions), and distances from c2 to all other points
        c2 = points.loc[dist_c1.idxmax()]
        dist_c2 = self._calc_distances_to_point(points, c2)

        # Get farthest atom to farthest atom to centroid c2 (in 4 dimensions), and distances from c3 to all other points
        c3 = points.loc[dist_c2.idxmax()]
        dist_c3 = self._calc_distances_to_point(points, c3)

        # Get forth and fifth reference point:
        # 1. Define two vectors spanning c1, c2, and c3 (from c1)
        a = c2 - c1
        b = c3 - c1

        # 2. Get only spatial part of a and b
        a_s = a[0:3]
        b_s = b[0:3]

        # 3. Calculate cross product of a_s and b_s
        # (keep same units and normalise vector to have half the norm of vector a)
        cross = np.cross(a_s, b_s)
        c_s = pd.Series(np.sqrt(sum(a ** 2)) / 2 * cross / (np.sqrt(sum(cross ** 2))), index=['x', 'y', 'z'])

        # 4. Add c to c1 to define the spatial components of the forth and fifth reference points

        # Add 4th dimension
        name_4thdim = points.columns[3]
        c = c_s.append(pd.Series([0], index=[name_4thdim]))
        c1_s = c1[0:3].append(pd.Series([0], index=[name_4thdim]))

        # Get values for 4th dimension (min and max of 4th dimension)
        max_value_4thdim = max(points.iloc[:, [3]].values)[0]
        min_value_4thdim = min(points.iloc[:, [3]].values)[0]

        c4 = c1_s + c + pd.Series([0, 0, 0, scaling_factor * max_value_4thdim], index=['x', 'y', 'z', name_4thdim])
        c5 = c1_s + c + pd.Series([0, 0, 0, scaling_factor * min_value_4thdim], index=['x', 'y', 'z', name_4thdim])

        # Get distances from c4 and c5 to all other points
        dist_c4 = self._calc_distances_to_point(points, c4)
        dist_c5 = self._calc_distances_to_point(points, c5)

        c = [c1, c2, c3, c4, c5]
        dist = [dist_c1, dist_c2, dist_c3, dist_c4, dist_c5]

        return self._get_shape_dict(c, dist)

    def _calc_shape_6dim(self, points, scaling_factor=1):
        """
        Encode binding site in 6D.

        Parameters
        ----------
        points : pandas.DataFrame
            Coordinates of representatives (points) in binding site.
        scaling_factor : int or float
            Scaling factor that can put higher or lower weight on non-spatial dimensions.

        Returns
        -------

        Notes
        -----
        1. Calculate reference points for a set of points:
           - c1, centroid
           - c2, closest atom to c1
           - c3, farthest atom to c1
           - c4, farthest atom to c3
           - c5, nearest atom to translated and scaled cross product of two vectors spanning c1, c2, and c3
           - c6, nearest atom to translated and scaled cross product of two vectors spanning c1, c3, and c4
           - c7, nearest atom to translated and scaled cross product of two vectors spanning c1, c4, and c2
        2. Calculate distances (distance distribution) from reference points to all other points.
        3. Calculate first, second, and third moment for each distance distribution.
        """

        if points.shape[1] != 6:
            sys.exit(f'Error: Dimension of input (points) is not {points.shape[1]} as requested.')

        # Get centroid of input coordinates (in 7 dimensions), and distances from c1 to all other points
        c1 = points.mean(axis=0)
        dist_c1 = self._calc_distances_to_point(points, c1)

        # Get closest and farthest atom to centroid c1 (in 7 dimensions),
        # and distances from c2 and c3 to all other points
        c2, c3 = points.loc[dist_c1.idxmin()], points.loc[dist_c1.idxmax()]
        dist_c2 = self._calc_distances_to_point(points, c2)
        dist_c3 = self._calc_distances_to_point(points, c3)

        # Get farthest atom to farthest atom to centroid c2 (in 7 dimensions),
        # and distances from c3 to all other points
        c4 = points.loc[dist_c3.idxmax()]
        dist_c4 = self._calc_distances_to_point(points, c4)

        # Get adjusted cross product
        c5_3d = self._calc_adjusted_3d_cross_product(c1, c2, c3)  # FIXME order of importance, right?
        c6_3d = self._calc_adjusted_3d_cross_product(c1, c3, c4)
        c7_3d = self._calc_adjusted_3d_cross_product(c1, c4, c2)

        # Get remaining reference points as nearest atoms to adjusted cross products
        c5 = self._calc_nearest_point(c5_3d, points, scaling_factor)
        c6 = self._calc_nearest_point(c6_3d, points, scaling_factor)
        c7 = self._calc_nearest_point(c7_3d, points, scaling_factor)

        # Get distances from c5, c6, and c7 to all other points
        dist_c5 = self._calc_distances_to_point(points, c5)
        dist_c6 = self._calc_distances_to_point(points, c6)
        dist_c7 = self._calc_distances_to_point(points, c7)

        c = [c1, c2, c3, c4, c5, c6, c7]
        dist = [dist_c1, dist_c2, dist_c3, dist_c4, dist_c5, dist_c6, dist_c7]

        return self._get_shape_dict(c, dist)

    @staticmethod
    def _calc_adjusted_3d_cross_product(coord_origin, coord_point_a, coord_point_b):
        """
        Calculates a translated and scaled 3D cross product vector based on three input vectors.

        Parameters
        ----------
        coord_origin : pandas.Series
            Point with a least N dimensions (N > 2).
        coord_point_a : pandas.Series
            Point with a least N dimensions (N > 2)
        coord_point_b : pandas.Series
            Point with a least N dimensions (N > 2)

        Returns
        -------
        pandas.Series
            Translated and scaled cross product 3D.

        Notes
        -----
        1. Generate two vectors based on the first three coordinates of the three input vectors (origin, point_a, and
           point_b), originating from origin.
        2. Calculate their cross product
           - scaled to length of the mean of both vectors and
           - translated to the origin.
        3. Get nearest point in points (in 3D) and return its 6D vector.
        """

        if not coord_origin.size == coord_point_a.size == coord_point_b.size:
            sys.exit('Error: The three input pandas.Series are not of same length.')
        if not coord_origin.size > 2:
            sys.exit('Error: The three input pandas.Series are not at least of length 3.')

        # Span vectors to point a and b from origin point
        a = coord_point_a - coord_origin
        b = coord_point_b - coord_origin

        # Get only first three dimensions
        a_3d = a[0:3]
        b_3d = b[0:3]

        # Calculate norm of vectors and their mean norm
        a_3d_norm = np.sqrt(sum(a_3d ** 2))
        b_3d_norm = np.sqrt(sum(b_3d ** 2))
        ab_3d_norm = (a_3d_norm + b_3d_norm) / 2

        # Calculate cross product
        cross = np.cross(a_3d, b_3d)

        # Calculate norm of cross product
        cross_norm = np.sqrt(sum(cross ** 2))

        # Calculate unit vector of cross product
        cross_unit = cross / cross_norm

        # Adjust cross product to length of the mean of both vectors described by the cross product
        cross_adjusted = ab_3d_norm * cross_unit

        # Cast adjusted cross product to Series
        coord_cross = pd.Series(cross_adjusted, index=a_3d.index)

        # Move adjusted cross product so that it originates from origin point
        c_3d = coord_origin[0:3] + coord_cross

        return c_3d

    def _calc_adjusted_6d_cross_product(self):
        pass

    def _get_shape_dict(self, ref_points, dist):
        """
        Get shape of binding site, i.e. reference points, distance distributions, and moments.

        Parameters
        ----------
        ref_points : list of pandas.Series
            Reference points (spatial and physicochemical properties).
        dist : list of pandas.Series
            Distances from each reference point to representatives (points).

        Returns
        -------
        dict of DataFrames
            Reference points, distance distributions, and moments.
        """

        # Store reference points as data frame
        ref_points = pd.concat(ref_points, axis=1).transpose()
        ref_points.index = ['c' + str(i) for i in range(1, len(ref_points.index) + 1)]

        # Store distance distributions as data frame
        dist = pd.concat(dist, axis=1)
        dist.columns = ['dist_c' + str(i) for i in range(1, len(dist.columns) + 1)]

        # Get first, second, and third moment for each distance distribution
        moments = self._calc_moments(dist)

        # Save shape as dictionary
        shape = {'ref_points': ref_points, 'dist': dist, 'moments': moments}

        return shape

    @staticmethod
    def _calc_distances_to_point(points, ref_point):
        """
        Calculate distances from one point (reference point) to all other points.

        Parameters
        ----------
        points : pandas.DataFrame
            Coordinates of representatives (points) in binding site.
        ref_point : pandas.Series
            Coordinates of one reference point.

        Returns
        -------
        pandas.Series
            Distances from reference point to representatives.
        """
        # TODO use here np.linalg.norm?
        distances = np.sqrt(((points - ref_point) ** 2).sum(axis=1))

        return distances

    def _calc_nearest_point(self, point, points, scaling_factor):
        """
        Get the point (N-dimensional) in a set of points that is nearest (in 3D) to an input point (3D).

        Parameters
        ----------
        point : pandas.Series
            Point in 3D
        points : DataFrame
            Points in at least 3D.

        Returns
        -------
        pandas.Series
            Point in **points** with all dimensions that is nearest in 3D to **point**.
        """

        if not point.size == 3:
            sys.exit('Error: Input point is not of length 3.')
        if not points.shape[1] > 2:
            sys.exit('Error: Input points are not of at least length 3.')

        # Get distances (in space, i.e. 3D) to all points
        dist_c_3d = self._calc_distances_to_point(points.iloc[:, 0:3], point)

        # Get nearest atom (in space, i.e. 3D) and save its 6D vector
        c_6d = points.loc[
            dist_c_3d.idxmin()]  # Note: idxmin() returns DataFrame index label (.loc) not position (.iloc)

        # Apply scaling factor on non-spatial dimensions
        scaling_vector = pd.Series([1] * 3 + [scaling_factor] * (points.shape[1] - 3), index=c_6d.index)
        c_6d = c_6d * scaling_vector

        return c_6d

    @staticmethod
    def _calc_moments(dist):
        """
        Calculate first, second, and third moment (mean, standard deviation, and skewness) for a distance distribution.

        Parameters
        ----------
        dist : pandas.Series
            Distance distribution, i.e. distances from reference point to all representatives (points)

        Returns
        -------
        pandas.Series
            First, second, and third moment of distance distribution.
        """

        # Get first, second, and third moment (mean, standard deviation, and skewness) for a distance distribution
        if len(dist) > 0:
            m1 = dist.mean()
            m2 = dist.std()
            m3 = pd.Series(cbrt(skew(dist.values, axis=0)), index=dist.columns.tolist())
        else:
            # In case there is only one data point.
            # However, this should not be possible due to restrictions in get_shape function.
            m1, m2, m3 = None, None, None

        # Store all moments in data frame
        moments = pd.concat([m1, m2, m3], axis=1)
        moments.columns = ['m1', 'm2', 'm3']

        return moments

    def _reorder_subset_keys(self):
        """
        Change the key order of the nested dictionary data_pseudocenter_subsets (Shapes attribute).
        Example: 'pc/z123/H/6Dratar1/moments' is changed to 'pc/z123/6Dratar1/H/moments'.

        Returns
        -------
        dict of pandas.DataFrames
            Dictionary of DataFrames.
        """

        self.data_pseudocenter_subsets = flatten(self.data_pseudocenter_subsets, reducer='path')

        keys_old = self.data_pseudocenter_subsets.keys()
        keys_new = [i.split('/') for i in keys_old]
        key_order = [0, 1, 3, 2, 4]
        keys_new = [[i[j] for j in key_order] for i in keys_new]
        keys_new = ['/'.join(i) for i in keys_new]

        for key_old, key_new in zip(keys_old, keys_new):
            self.data_pseudocenter_subsets[key_new] = self.data_pseudocenter_subsets.pop(key_old)

        return self.data_pseudocenter_subsets


def process_encoding(input_mol_path, output_dir):
    """
    Process a list of molecule structure files (retrieved by an input path to one or multiple files) and
    save per binding site multiple output files to an output directory.

    Each binding site is processed as follows:
      * Create all necessary output directories and sets all necessary file paths.
      * Encode the binding site.
      * Save the encoded binding sites as pickle file.
      * Save the reference points as PyMol cgo file.

    The output file systems is constructed as follows:

    output_dir/
      encoding/
        molecule_id_1/
          ratar_encoding.p
          ref_points_cgo.py
        molecule_id_2/
          ...
      ratar.log

    Parameters
    ----------
    input_mol_path : str
        Path to molecule structure file(s), can include a wildcard to match multiple files.
    output_dir : str
        Output directory.
    """

    # Get all molecule structure files
    input_mol_path_list = glob.glob(input_mol_path)
    input_mol_path_list = input_mol_path_list

    if len(input_mol_path_list) == 0:
        logger.info(f'Input path matches no molecule files: {input_mol_path}', extra={'molecule_id': 'all'})
    else:
        logger.info(f'Input path matches {len(input_mol_path_list)} molecule file(s): {input_mol_path}',
                    extra={'molecule_id': 'all'})

    # Get number of molecule structure files and set molecule structure counter
    mol_sum = len(input_mol_path_list)

    # Iterate over all binding sites (molecule structure files)
    for mol_counter, mol in enumerate(input_mol_path_list, 1):

        # Load binding site from molecule structure file
        bs_loader = MoleculeLoader(mol)
        bs_loader.load_molecule(remove_solvent=True)
        pmols = bs_loader.pmols

        # Get number of pmol objects and set pmol counter
        pmol_sum = len(pmols)

        # Iterate over all binding sites in molecule structure file
        for pmol_counter, pmol in enumerate(pmols, 1):

            # Get iteration progress
            logger.info(f'Encoding: {mol_counter}/{mol_sum} molecule structure file - {pmol_counter}/{pmol_sum} molecule',
                        extra={'molecule_id': pmol.code})

            # Process single binding site:

            # Create output folder
            molecule_id_encoding = Path(output_dir) / 'encoding' / pmol.code
            create_directory(molecule_id_encoding)

            # Get output file paths
            output_enc_path = molecule_id_encoding / 'ratar_encoding.p'
            output_cgo_path = molecule_id_encoding / 'ref_points_cgo.py'

            # Encode binding site
            binding_site = BindingSite(pmol)

            # Save binding site
            save_binding_site(binding_site, str(output_enc_path))

            # Save binding site reference points as cgo file
            save_cgo_file(binding_site, str(output_cgo_path))


def save_binding_site(binding_site, output_path):
    """
    Save an encoded binding site to a pickle file in an output directory.

    Parameters
    ----------
    binding_site : encoding.BindingSite
        Encoded binding site.
    output_path : str
        Path to output file.
    """

    create_directory(Path(output_path).parent)

    with open(output_path, 'wb') as f:
        pickle.dump(binding_site, f)


def save_cgo_file(binding_site, output_path):
    """
    Generate a CGO file containing reference points for different encoding methods.

    Parameters
    ----------
    binding_site : encoding.BindingSite
        Encoded binding site.
    output_path : str
        Path to output file.

    Notes
    -----
    Python script (cgo file) for PyMol.
    """

    # Set PyMol sphere colors (for reference points)
    sphere_colors = sns.color_palette('hls', 7)

    # List contains lines for python script
    lines = [
        'from pymol import *',
        'import os',
        'from pymol.cgo import *',
        ''
    ]
    # Collect all PyMol objects here (in order to group them after loading them to PyMol)
    obj_names = []

    # Flatten nested dictionary
    bs_flat = flatten(binding_site.shapes.data, reducer='path')

    # Select keys for reference points
    bs_flat_keys = [i for i in bs_flat.keys() if 'ref_points' in i]

    for key in bs_flat_keys:

        if bs_flat[key] is None:
            logger.info(f'Empty encoding for {key}.')

        else:

            # Get reference points (coordinates)
            ref_points = bs_flat[key]

            # Set descriptive name for reference points (PDB ID, representatives, dimensions, encoding method)
            obj_name = f'{key.replace("/", "_").replace("_ref_points", "")}__{binding_site.molecule_id}'
            obj_names.append(obj_name)

            # Set size for PyMol spheres
            size = str(1)

            lines.append(f'obj_{obj_name} = [')  # Variable cannot start with digit, thus add prefix obj_

            # Set color counter (since we iterate over colors for each reference point)
            counter_colors = 0

            # For each reference point, write sphere color, coordinates and size to file
            for index, row in ref_points.iterrows():

                # Set sphere color
                sphere_color = list(sphere_colors[counter_colors])
                counter_colors = counter_colors + 1

                # Write sphere a) color and b) coordinates and size to file
                lines.extend(
                    [
                        f'\tCOLOR, {str(sphere_color[0])}, {str(sphere_color[1])}, {str(sphere_color[2])},',
                        f'\tSPHERE, {str(row["x"])}, {str(row["y"])}, {str(row["z"])}, {size},'
                    ]
                )

            # Write command to file that will load the reference points as PyMol object
            lines.extend(
                [
                    f']',
                    f'cmd.load_cgo(obj_{obj_name}, "{obj_name}")',
                    ''
                ]
            )

    # Group all objects to one group
    lines.append(f'cmd.group("{binding_site.molecule_id[:4]}_ref_points", "{" ".join(obj_names)}")')

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))


def get_encoded_binding_site_path(code, output_path):
    """
    Get a binding site pickle path based on a path wildcard constructed from
    - a molecule code (can contain only PDB ID to check for file matches) and
    - the path to the ratar encoding output directory.

    Constructed wildcard: f'{output_path}/encoding/*{code}*/ratar_encoding.p'

    Parameters
    ----------
    code: str
        Molecule code (is or contains PDB ID).
    output_path: str
        Path to output directory containing ratar related files.

    Returns
    -------
    str
        Path to binding site pickle file.
    """

    # Define wildcard for path to pickle file
    bs_wildcard = f'{output_path}/encoding/*{code}*/ratar_encoding.p'

    # Retrieve all paths that match the wildcard
    bs_path = glob.glob(bs_wildcard)

    # If wildcard matches no file, retry.
    if len(bs_path) == 0:
        lines = [
            'Error: Your input path matches no file. Please retry.',
            'Your input wildcard was the following: ',
            bs_wildcard
        ]
        raise FileNotFoundError('\n'.join(lines))

    # If wildcard matches multiple files, retry.
    elif len(bs_path) > 1:
        lines = [
            'Error: Your input path matches multiple files. Please select one of the following as input string: ',
            bs_path,
            'Your input wildcard was the following: ',
            bs_wildcard
        ]
        raise FileNotFoundError('\n'.join(lines))

    # If wildcard matches one file, return file path.
    else:
        bs_path = bs_path[0]
        return bs_path


def load_binding_site(output_path):
    """
    Load an encoded binding site from a pickle file.

    Parameters
    ----------
    output_path : str
        Path to binding site pickle file.

    Returns
    -------
    encoding.BindingSite
        Encoded binding site.
    """

    # Retrieve all paths that match the input path
    bs_path = glob.glob(output_path)

    # If input path matches no file, retry.
    if len(bs_path) == 0:
        lines = [
            'Error: Your input path matches no file. Please retry.',
            'Your input path was the following: ',
            bs_path
        ]
        raise FileNotFoundError('\n'.join(lines))

    # If input path matches multiple files, retry.
    elif len(bs_path) > 1:
        lines = [
            'Error: Your input path matches multiple files. Please select one of the following as input string: ',
            bs_path,
            '\nYour input path was the following: ',
            output_path
        ]
        raise FileNotFoundError('\n'.join(lines))

    # If input path matches one file, load file.
    else:
        bs_path = bs_path[0]
        with open(bs_path, 'rb') as f:
            binding_site = pickle.load(f)
        return binding_site
