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
import warnings

from flatten_dict import flatten, unflatten
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import cbrt
from scipy.stats.stats import moment

from ratar.auxiliary import MoleculeLoader, AminoAcidDescriptors
from ratar.auxiliary import create_directory, load_pseudocenters

warnings.simplefilter('error', FutureWarning)

logger = logging.getLogger(__name__)


class BindingSite:
    """
    Class used to represent a molecule and its encoding.

    Parameters
    ----------
    molecule : biopandas.mol2.pandas_mol2.PandasMol2 or biopandas.pdb.pandas_pdb.PandasPdb
        Content of mol2 or pdb file as BioPandas object.

    Attributes
    ----------
    molecule : biopandas.mol2.pandas_mol2.PandasMol2 or biopandas.pdb.pandas_pdb.PandasPdb
            Content of mol2 or pdb file as BioPandas object.
    representatives : ratar.encoding.Representatives
        Representative atoms of binding site for different representation methods.
    shapes : Shapes
        Encoded binding site (reference points, distance distribution and distribution moments).

    Examples
    --------
    >>> from ratar.auxiliary import MoleculeLoader
    >>> from ratar.encoding import Representatives, Coordinates

     >>> molecule_path = Path(__name__).parent / 'ratar' / 'tests' / 'data' / 'AAK1_4wsq_altA_chainA.mol2'

    >>> molecule_loader = MoleculeLoader(molecule_path, remove_solvent=True)
    >>> molecule = molecule_loader.molecules[0]

    >>> binding_site = BindingSite(molecule)
    """

    def __init__(self, molecule):

        self.molecule = molecule
        self.representatives = self.get_representatives()
        self.shapes = self.run()

    def __eq__(self, other):
        """
        Check if two BindingSite objects are equal.

        Returns
        -------
        bool
            True if molecule ID, molecule DataFrames, Representative object, and Shapes object are equal, else False.
        """

        rules = [
            self.molecule.code == other.molecule.code,
            self.molecule.df.equals(other.molecule.df),
            self.representatives == other.representatives,
            self.shapes == other.shapes
        ]
        return all(rules)

    def get_representatives(self):
        """
        Get representatives of a molecule.

        Returns
        -------
        ratar.encoding.Representatives
            Different representatives (representative atoms) of a molecule.
        """

        representatives = Representatives()
        representatives.from_molecule(self.molecule)
        return representatives

    @staticmethod
    def get_coordinates(representatives):
        """
        Get coordinates for different representatives of a molecule.

        Parameters
        ----------
        representatives : ratar.encoding.Representatives
            Different representatives (representative atoms) of a molecule.

        Returns
        -------
        ratar.encoding.Coordinates
            Coordinates for different representatives of a molecule.
        """

        coordinates = Coordinates()
        coordinates.from_representatives(representatives)
        return coordinates

    @staticmethod
    def get_physicochemicalproperties(representatives):
        """
        Get different physicochemical properties for different representatives of a molecule.

        Parameters
        ----------
        representatives : ratar.encoding.Representatives
            Different representatives (representative atoms) of a molecule.

        Returns
        -------
        ratar.encoding.PhysicochemicalProperties
            Different physicochemical properties for different representatives of a molecule.
        """

        physicochemicalproperties = PhysicoChemicalProperties()
        physicochemicalproperties.from_representatives(representatives)
        return physicochemicalproperties

    @staticmethod
    def get_subsets(representatives):
        """
        Get subsets (atom indices) for different representatives of a molecule.

        Parameters
        ----------
        representatives : ratar.encoding.Representatives
            Different representatives (representative atoms) of a molecule.

        Returns
        -------
        ratar.encoding.Subsets
            Subsets (atom indices) for different representatives of a molecule.
        """

        subsets = Subsets()
        subsets.from_representatives(representatives)
        return subsets

    def get_points(self, coordinates, physicochemicalproperties, subsets):
        """
        Get multidimensional points for different representatives of a molecule, which contain information on
        coordinates (spatial dimensions) and physicochemical properties (physicochemical dimensions).
        Additionally, group points into subsets.

        Parameters
        ----------
        coordinates : ratar.encoding.Coordinates
            Coordinates for different representatives of a molecule.
        physicochemicalproperties : ratar.encoding.PhysicochemicalProperties
            Different physicochemical properties for different representatives of a molecule.
        subsets : ratar.encoding.Subsets
            Subsets (atom indices) for different representatives of a molecule.

        Returns
        -------
        ratar.encoding.Points
            Multidimensional points for different representatives of a molecule, which contain information on
            coordinates (spatial dimensions) and physicochemical properties (physicochemical dimensions).
            Additionally, points are grouped into subsets.
        """

        points = Points()
        points.from_properties(coordinates, physicochemicalproperties)
        points.from_subsets(subsets)
        return points

    def get_shapes(self, points):
        """
        Get different shape encodings for points representing a molecule.

        Parameters
        ----------
        points : ratar.encoding.Points
            Multidimensional points for different representatives of a molecule, which contain information on
            coordinates (spatial dimensions) and physicochemical properties (physicochemical dimensions).
            Additionally, points are grouped into subsets.

        Returns
        -------
        ratar.encoding.Shapes
            Different shape encodings for different points representing a molecule.
        """

        shapes = Shapes()
        shapes.from_points(points)
        shapes.from_subset_points(points)
        return shapes

    def run(self):
        """
        Run shape encoding procedure for a molecule.

        Returns
        -------
        ratar.encoding.Shapes
            Different shape encodings for different points representing a molecule.
        """

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

    >>> molecule_path = Path(__name__).parent / 'ratar' / 'tests' / 'data' / 'AAK1_4wsq_altA_chainA.mol2'

    >>> molecule_loader = MoleculeLoader(molecule_path, remove_solvent=True)
    >>> molecule = molecule_loader.molecules[0]

    >>> representatives = Representatives()
    >>> representatives.from_molecule(molecule)
    >>> representatives
    """

    def __init__(self):

        self.molecule_id = ''
        self.data = {
            'ca':  pd.DataFrame(),
            'pca':  pd.DataFrame(),
            'pc':  pd.DataFrame(),
        }

    @property
    def ca(self):
        """
        Get data (as listed in structural file) for representatives: Calpha.

        Returns
        -------
        pandas.DataFrame
            Data (as listed in structural file) for representatives: Calpha.
        """

        return self.data['ca']

    @property
    def pca(self):
        """
        Get data (as listed in structural file) for representatives: pseudocenter atoms.

        Returns
        -------
        pandas.DataFrame
            Data (as listed in structural file) for representatives: pseudocenter atoms.
        """

        return self.data['pca']

    @property
    def pc(self):
        """
        Get data (as listed in structural file) for representatives: pseudocenters (consisting of pseudocenter atoms).

        Returns
        -------
        pandas.DataFrame
            Data (as listed in structural file) for representatives: pseudocenters (consisting of pseudocenter atoms).
        """

        return self.data['pc']

    def __eq__(self, other):
        """
        Check if two Representatives objects are equal.

        Returns
        -------
        bool
            True if dictionary keys (strings) and dictionary values (DataFrames) are equal, else False.
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

    def from_molecule(self, molecule):
        """
        Set molecule ID and extract binding site representatives.

        Parameters
        ----------
        molecule : biopandas.mol2.pandas_mol2.PandasMol2 or biopandas.pdb.pandas_pdb.PandasPdb
            Content of mol2 or pdb file as BioPandas object.
        """

        self.molecule_id = molecule.code

        for key in self.data.keys():
            if key == 'ca':
                self.data['ca'] = self._get_ca(molecule.df)
            elif key == 'pca':
                self.data['pca'] = self._get_pca(molecule.df)
            elif key == 'pc':
                self.data['pc'] = self._get_pc(molecule.df)
            else:
                raise KeyError(f'Unknown representative key: {key}')

    @staticmethod
    def _get_ca(molecule_df):
        """
        Extract Calpha atoms from binding site.

        Parameters
        ----------
        molecule_df : pandas.DataFrame
            DataFrame containing atom lines from input file.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing atom lines from input file described by Z-scales.
        """

        # Filter for atom name 'CA' (first condition) but exclude calcium (second condition)
        molecule_ca = molecule_df[(molecule_df['atom_name'] == 'CA') & (molecule_df['res_name'] != 'CA')]

        return molecule_ca

    @staticmethod
    def _get_pca(molecule_df):
        """
        Extract pseudocenter atoms from molecule.

        Parameters
        ----------
        molecule_df: pandas.DataFrame
            DataFrame containing atom lines from input file.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing atom lines from input file that belong to pseudocenters.
        """

        # Load pseudocenter atoms
        pseudocenter_atoms = load_pseudocenters()

        # Collect all molecule atoms that belong to pseudocenters
        molecule_pca = []

        for index, row in molecule_df.iterrows():

            query = f'{row["res_name"]}_{row["atom_name"]}'

            if query in list(pseudocenter_atoms['pc_atom_pattern']):  # Non-peptide bond atoms

                pc_ix = pseudocenter_atoms.index[pseudocenter_atoms['pc_atom_pattern'] == query].tolist()[0]

                molecule_pca.append(
                    [
                        index,
                        pseudocenter_atoms.iloc[pc_ix]['pc_type'],
                        pseudocenter_atoms.iloc[pc_ix]['pc_id'],
                        pseudocenter_atoms.iloc[pc_ix]['pc_atom_id']
                    ]
                )

            elif row['atom_name'] == 'O':  # Peptide bond atoms

                molecule_pca.append(
                    [
                        index,
                        'HBA',
                        'PEP_HBA_1',
                        'PEP_HBA_1_0'
                    ]
                )

            elif row['atom_name'] == 'N':  # Peptide bond atoms

                molecule_pca.append(
                    [
                        index,
                        'HBD',
                        'PEP_HBD_1',
                        'PEP_HBD_1_N']
                )

            elif row['atom_name'] == 'C':  # Peptide bond atoms

                molecule_pca.append(
                    [
                        index,
                        'AR',
                        'PEP_AR_1',
                        'PEP_AR_1_C'
                    ]
                )

        # Cast list of lists to DataFrame
        molecule_pca_df = pd.DataFrame(molecule_pca)
        molecule_pca_df.columns = ['index', 'pc_type', 'pc_id', 'pc_atom_id']
        molecule_pca_df.index = [i[0] for i in molecule_pca]
        molecule_pca_df.drop(columns=['index'], inplace=True)

        # Join molecule with pseudocenter subset
        molecule_pca_df = molecule_df.join(molecule_pca_df, how='outer')
        molecule_pca_df.dropna(how='any', inplace=True)

        return molecule_pca_df

    def _get_pc(self, molecule_df):
        """
        Extract pseudocenters from molecule.

        Parameters
        ----------
        molecule_df : pandas.DataFrame
            DataFrame containing atom lines from input file.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing atom lines for (aggregated) pseudocenters, i.e. aggregate multiple atoms belonging to
            one pseudocenter.
        """

        # Get pseudocenter atoms
        molecule_pca_df = self._get_pca(molecule_df)

        # Calculate pseudocenters
        molecule_pc = []

        for key, group in molecule_pca_df.groupby(['subst_name', 'pc_id'], sort=False):

            if len(group) == 1:  # If pseudocenter only contains one atom, keep data
                row = group.iloc[0].copy()
                row['atom_id'] = [row['atom_id']]
                row['atom_name'] = [row['atom_name']]

                molecule_pc.append(row)

            else:  # If pseudocenter contains multiple atoms, aggregate data
                first_row = group.iloc[0].copy()
                first_row['atom_id'] = list(group['atom_id'])
                first_row['atom_name'] = list(group['atom_name'])
                first_row['x'] = group['x'].mean()
                first_row['y'] = group['y'].mean()
                first_row['z'] = group['z'].mean()
                first_row['charge'] = group['charge'].mean()  # TODO Alternatives to mean of a charge?

                molecule_pc.append(first_row)

        molecule_pc_df = pd.concat(molecule_pc, axis=1).T

        # Change datatype from object to float for selected columns
        molecule_pc_df[['x', 'y', 'z', 'charge']] = molecule_pc_df[['x', 'y', 'z', 'charge']].astype(float)

        return molecule_pc_df


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
    >>> from ratar.encoding import Coordinates

     >>> molecule_path = Path(__name__).parent / 'ratar' / 'tests' / 'data' / 'AAK1_4wsq_altA_chainA.mol2'

    >>> molecule_loader = MoleculeLoader(molecule_path, remove_solvent=True)
    >>> molecule = molecule_loader.molecules[0]

    >>> coordinates = Coordinates()
    >>> coordinates.from_molecule(molecule)
    >>> coordinates
    """

    def __init__(self):

        self.molecule_id = ''
        self.data = {
            'ca': None,
            'pca': None,
            'pc': None,
        }

    @property
    def ca(self):
        """
        Get coordinates for representatives: Calpha.

        Returns
        -------
        pandas.DataFrame
            Coordinates for representatives: Calpha.
        """

        return self.data['ca']

    @property
    def pca(self):
        """
        Get coordinates for representatives: pseudocenter atoms.

        Returns
        -------
        pandas.DataFrame
            Coordinates for representatives: pseudocenter atoms.
        """

        return self.data['pca']

    @property
    def pc(self):
        """
        Get coordinates for representatives: pseudocenters (consisting of pseudocenter atoms).

        Returns
        -------
        pandas.DataFrame
            Coordinates for representatives: pseudocenters (consisting of pseudocenter atoms).
        """

        return self.data['pc']

    def __eq__(self, other):
        """
        Check if two Coordinates objects are equal.

        Returns
        -------
        bool
            True if dictionary keys (strings) and dictionary values (DataFrames) are equal, else False.
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

    def from_molecule(self, molecule):
        """
        Convenience class method: Get coordinates from molecule object.

        Parameters
        ----------
        molecule : biopandas.mol2.pandas_mol2.PandasMol2 or biopandas.pdb.pandas_pdb.PandasPdb
            Content of mol2 or pdb file as BioPandas object.
        """

        representatives = Representatives()
        representatives.from_molecule(molecule)

        self.from_representatives(representatives)

    def from_representatives(self, representatives):
        """
        Get coordinates (x, y, z) for molecule representatives.

        Parameters
        ----------
        representatives : ratar.encoding.Representatives
            Representatives class instance.

        Returns
        -------
        dict of DataFrames
            Dictionary (representatives types, e.g. 'pc') of DataFrames containing molecule coordinates.
        """

        self.molecule_id = representatives.molecule_id

        self.data = {}

        for k1, v1 in representatives.data.items():
            if isinstance(v1, pd.DataFrame):
                self.data[k1] = v1[['x', 'y', 'z']]
            elif isinstance(v1, dict):
                self.data[k1] = {k2: v2[['x', 'y', 'z']] for (k2, v2) in v1.items()}
            else:
                raise TypeError(f'Expected dict or pandas.DataFrame but got {type(v1)}')

        return self.data  # Return not necessary here, keep for clarity.


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
    >>> from ratar.encoding import PhysicoChemicalProperties

     >>> molecule_path = Path(__name__).parent / 'ratar' / 'tests' / 'data' / 'AAK1_4wsq_altA_chainA.mol2'

    >>> molecule_loader = MoleculeLoader(molecule_path, remove_solvent=True)
    >>> molecule = molecule_loader.molecules[0]

    >>> physicochemicalproperties = PhysicoChemicalProperties()
    >>> physicochemicalproperties.from_molecule(molecule)
    >>> physicochemicalproperties
    """

    def __init__(self):

        self.molecule_id = ''
        self.data = {
            'ca': {},
            'pca': {},
            'pc': {},
        }

    @property
    def ca(self):
        """
        Get physicochemical properties for representatives: Calpha.

        Returns
        -------
        dict of pandas.DataFrame
            Different physicochemical properties for representatives: Calpha.
        """

        return self.data['ca']

    @property
    def pca(self):
        """
        Get physicochemical properties for representatives: pseudocenter atoms.

        Returns
        -------
        dict of pandas.DataFrame
            Different physicochemical properties for representatives: pseudocenter atoms.
        """

        return self.data['pca']

    @property
    def pc(self):
        """
        Get physicochemical properties for representatives: pseudocenters (consisting of pseudocenter atoms).

        Returns
        -------
        dict of pandas.DataFrame
            Different physicochemical properties for representatives: pseudocenters (consisting of pseudocenter atoms).
        """

        return self.data['pc']

    def __eq__(self, other):
        """
        Check if two PhysicoChemicalProperties objects are equal.

        Returns
        -------
        bool
            True if dictionary keys (strings) and dictionary values (DataFrames) are equal, else False.
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

    def from_molecule(self, molecule):
        """
        Convenience class method: Get physicochemical properties from molecule object.

        Parameters
        ----------
        molecule : biopandas.mol2.pandas_mol2.PandasMol2 or biopandas.pdb.pandas_pdb.PandasPdb
            Content of mol2 or pdb file as BioPandas object.
        """

        representatives = Representatives()
        representatives.from_molecule(molecule)

        self.from_representatives(representatives)

    def from_representatives(self, representatives):
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

        self.molecule_id = representatives.molecule_id

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

        return self.data   # Return not necessary here, keep for clarity.

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
    >>> from ratar.encoding import Subsets

     >>> molecule_path = Path(__name__).parent / 'ratar' / 'tests' / 'data' / 'AAK1_4wsq_altA_chainA.mol2'

    >>> molecule_loader = MoleculeLoader(molecule_path, remove_solvent=True)
    >>> molecule = molecule_loader.molecules[0]

    >>> subsets = Subsets()
    >>> subsets.from_molecule(molecule)
    >>> subsets
    """

    def __init__(self):

        self.molecule_id = ''
        self.data_pseudocenter_subsets = {
            'pca': {},
            'pc': {}
        }

    @property
    def pseudocenters(self):
        """
        Get subset indices for different subsets based on pseudocenters (consisting of pseudocenter atoms).

        Returns
        -------
        Dict of lists
            Subset indices for different subsets based on pseudocenters (consisting of pseudocenter atoms).
        """

        return self.data_pseudocenter_subsets['pc']

    @property
    def pseudocenter_atoms(self):
        """
        Get subset indices for different subsets based on pseudocenter atoms.

        Returns
        -------
        Dict of lists
            Subset indices for different subsets based on pseudocenter atoms.
        """

        return self.data_pseudocenter_subsets['pca']

    def __eq__(self, other):
        """
        Check if two Subsets objects are equal.

        Returns
        -------
        bool
            True if dictionary keys (strings) and dictionary values (DataFrames) are equal, else False.
        """

        obj1 = flatten(self.data_pseudocenter_subsets, reducer='path')
        obj2 = flatten(other.data_pseudocenter_subsets, reducer='path')

        try:
            rules = [
                obj1.keys() == obj2.keys(),
                all([v == obj2[k] for k, v in obj1.items()])
            ]
        except KeyError:
            rules = False

        return all(rules)

    def from_molecule(self, molecule):
        """
        Convenience class method: Get subset indices from molecule object.

        Parameters
        ----------
        molecule : biopandas.mol2.pandas_mol2.PandasMol2 or biopandas.pdb.pandas_pdb.PandasPdb
            Content of mol2 or pdb file as BioPandas object.
        """

        representatives = Representatives()
        representatives.from_molecule(molecule)

        self.from_representatives(representatives)

    def from_representatives(self, representatives):
        """
        Extract feature subsets from e.g. pseudocenters (pseudocenter atoms).

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

        self.molecule_id = representatives.molecule_id

        # Subset: pseudocenter atoms
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

        return self.data_pseudocenter_subsets   # Return not necessary here, keep for clarity.


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
    >>> from ratar.encoding import Points

     >>> molecule_path = Path(__name__).parent / 'ratar' / 'tests' / 'data' / 'AAK1_4wsq_altA_chainA.mol2'

    >>> molecule_loader = MoleculeLoader(molecule_path, remove_solvent=True)
    >>> molecule = molecule_loader.molecules[0]

    >>> points = Points()
    >>> points.from_molecule(molecule)
    >>> points
    """

    def __init__(self):

        self.molecule_id = ''
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
        """
        Get multidimensional points, consisting of coordinates and physicochemical properties, for representatives:
        Calpha.

        Returns
        -------
        dict of pandas.DataFrame
            Multidimensional points for representatives for different physicochemical properties: Calpha.
        """

        return self.data['ca']

    @property
    def pca(self):
        """
        Get multidimensional points, consisting of coordinates and physicochemical properties, for representatives:
        pseudocenter atoms.

        Returns
        -------
        dict of pandas.DataFrame
            Multidimensional points for representatives for different physicochemical properties: pseudocenter atoms.
        """

        return self.data['pca']

    @property
    def pc(self):
        """
        Get multidimensional points, consisting of coordinates and physicochemical properties, for representatives:
        pseudocenters (consisting of pseudocenter atoms).

        Returns
        -------
        dict of dict of pandas.DataFrame
            Multidimensional points for representatives for different physicochemical properties: pseudocenters
            (consisting of pseudocenter atoms).
        """

        return self.data['pc']

    @property
    def pca_subsets(self):
        """
        Get multidimensional points, consisting of coordinates and physicochemical properties, for representatives:
        subsets of pseudocenters (consisting of pseudocenter atoms).

        Returns
        -------
        dict of dict of pandas.DataFrame
            Multidimensional points for representatives for different physicochemical properties and for different
            subsets: pseudocenters (consisting of pseudocenter atoms).
        """

        return self.data_pseudocenter_subsets['pca']

    @property
    def pc_subsets(self):
        """
        Get multidimensional points, consisting of coordinates and physicochemical properties, for representatives:
        subsets of pseudocenter atoms.

        Returns
        -------
        dict of dict of pandas.DataFrame
            Multidimensional points for representatives for different physicochemical properties and for different
            subsets: pseudocenter atoms.
        """

        return self.data_pseudocenter_subsets['pc']

    def __eq__(self, other):
        """
        Check if two Points objects are equal.

        Returns
        -------
        bool
            True if dictionary keys (strings) and dictionary values (DataFrames) are equal, else False.
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

    def from_molecule(self, molecule):
        """
        Convenience class method: Get points from molecule object.

        Parameters
        ----------
        molecule : biopandas.mol2.pandas_mol2.PandasMol2 or biopandas.pdb.pandas_pdb.PandasPdb
            Content of mol2 or pdb file as BioPandas object.
        """

        representatives = Representatives()
        representatives.from_molecule(molecule)

        coordinates = Coordinates()
        coordinates.from_representatives(representatives)

        physicochemicalproperties = PhysicoChemicalProperties()
        physicochemicalproperties.from_representatives(representatives)

        subsets = Subsets()
        subsets.from_representatives(representatives)

        self.from_properties(coordinates, physicochemicalproperties)
        self.from_subsets(subsets)

    def from_properties(self, coordinates, physicochemicalproperties):
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

        self.molecule_id = coordinates.molecule_id

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

        return self.data   # Return not necessary here, keep for clarity.

    def from_subsets(self, subsets):
        """
        Group points into subsets, e.g. subsets by pseudocenter types.

        Parameters
        ----------
        subsets : ratar.encoding.Subsets
            Subsets class instance.

        Returns
        -------
        dict of dict of dict of pandas.DataFrames
            Dictionary (representatives types, e.g. 'pc') of dictionaries (physicochemical properties, e.g. 'pc_z123')
        """

        # Subsets can only be generated if points are already created, so make check:
        if not self.data["ca"]:
            raise TypeError("Attribute data of Points class is empty. Before you can generate subsets, you need to set "
                            "the points to be subset. Use Points.from_properties() class method.")

        # Subset: pseudocenter atoms
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

        return self.data_pseudocenter_subsets   # Return not necessary here, keep for clarity.


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
    >>> from ratar.encoding import Shapes

     >>> molecule_path = Path(__name__).parent / 'ratar' / 'tests' / 'data' / 'AAK1_4wsq_altA_chainA.mol2'

    >>> molecule_loader = MoleculeLoader(molecule_path, remove_solvent=True)
    >>> molecule = molecule_loader.molecules[0]

    >>> shapes = Shapes()
    >>> shapes.from_molecule(molecule)
    >>> shapes
    """

    def __init__(self):

        self.molecule_id = ''
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
        """
        Get different shape encodings for representatives: Calpha.

        Returns
        -------
        dict of dict of dict of pandas.DataFrame
            Different shape encodings for representatives for different physicochemical properties: Calpha.
        """

        return self.data['ca']

    @property
    def pca(self):
        """
        Get different shape encodings for representatives: pseudocenter atoms.

        Returns
        -------
        dict of dict of dict of pandas.DataFrame
            Different shape encodings for representatives for different physicochemical properties: pseudocenter atoms.
        """

        return self.data['pca']

    @property
    def pc(self):
        """
        Get different shape encodings for representatives: pseudocenters (consisting of pseudocenter atoms).

        Returns
        -------
        dict of dict of dict of pandas.DataFrame
            Different shape encodings for representatives for different physicochemical properties: pseudocenters
            (consisting of pseudocenter atoms).
        """

        return self.data['pc']

    @property
    def pca_subsets(self):
        """
        Get shape encodings for representatives: subsets of pseudocenters (consisting of pseudocenter atoms).

        Returns
        -------
        dict of dict of dict of pandas.DataFrame
            Different shape encodings for different physicochemical properties and for different
            subsets: pseudocenter atoms.
        """

        return self.data_pseudocenter_subsets['pca']

    @property
    def pc_subsets(self):
        """
        Get shape encodings for representatives: subsets of pseudocenters (consisting of pseudocenter atoms).

        Returns
        -------
        dict of dict of dict of pandas.DataFrame
            Different shape encodings for different physicochemical properties and for different
            subsets: pseudocenters (consisting of pseudocenter atoms).
        """

        return self.data_pseudocenter_subsets['pc']

    def __eq__(self, other):
        """
        Check if two Shapes objects are equal.

        Returns
        -------
        bool
            True if dictionary keys (strings) and dictionary values (DataFrames) are equal, else False.
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

    def from_molecule(self, molecule):
        """
        Convenience class method: Get shapes from molecule object.

        Parameters
        ----------
        molecule : biopandas.mol2.pandas_mol2.PandasMol2 or biopandas.pdb.pandas_pdb.PandasPdb
        Content of mol2 or pdb file as BioPandas object.
        """

        representatives = Representatives()
        representatives.from_molecule(molecule)

        coordinates = Coordinates()
        coordinates.from_representatives(representatives)

        physicochemicalproperties = PhysicoChemicalProperties()
        physicochemicalproperties.from_representatives(representatives)

        subsets = Subsets()
        subsets.from_representatives(representatives)

        points = Points()
        points.from_properties(coordinates, physicochemicalproperties)
        points.from_subsets(subsets)

        self.from_points(points)
        self.from_subset_points(points)

    def from_points(self, points):
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

        # Flatten nested dictionary
        points_flat = flatten(points.data, reducer='path')

        for k, v in points_flat.items():
            self.data[k] = self._get_shape_by_method(v)

        # Unflatten dictionary back to nested dictionary
        self.data = unflatten(self.data, splitter='path')

        return self.data   # Return not necessary here, keep for clarity.

    def from_subset_points(self, points):
        """
        Get the encoding of a subset molecule for different types of representatives, physicochemical properties, and encoding
        methods.

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

        # Flatten nested dictionary
        points_flat = flatten(points.data_pseudocenter_subsets, reducer='path')

        for k, v in points_flat.items():
            self.data_pseudocenter_subsets[k] = self._get_shape_by_method(v, k)

        # Change key order of (flattened) nested dictionary (reverse subset type and encoding type)
        # Example: 'pc/z123/H/6Dratar1/moments' is changed to 'pc/z123/6Dratar1/H/moments'.
        self.data_pseudocenter_subsets = self._reorder_nested_dict_keys(self.data_pseudocenter_subsets, [0, 1, 3, 2, 4])

        # Unflatten dictionary back to nested dictionary
        self.data_pseudocenter_subsets = unflatten(self.data_pseudocenter_subsets, splitter='path')

        return self.data_pseudocenter_subsets   # Return not necessary here, keep for clarity.

    def _get_shape_by_method(self, points_df, points_key=None):
        """
        Apply encoding method on points depending on points dimensions and return encoding.

        Parameters
        ----------
        points_df : pandas.DataFrame
            DataFrame containing points which can have different dimensions.
        points_key : str
            String describing the type of representatives, physicochemical properties, and subsets that the points are
            based on.

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
            return {'6Dratar1': self._calc_shape_6dim_ratar1(points_df)}
        elif n_dimensions < 3:
            logger.warning(f'{points_key}: Unexpected points dimension: {points_df.shape[1]}. Not implemented.')
            return {'encoding_failed': {'dist': None, 'ref_points': None, 'moments': None}}
            # raise ValueError(f'Unexpected points dimension: {points_df.shape[1]}. Not implemented.')
        elif n_dimensions == 3 and n_points <= 3:
            logger.warning(f'{points_key}: Number of points in 3D must be at least 4. Number of input points: '
                           f'{points_df.shape[0]}.')
            return {'encoding_failed': {'dist': None, 'ref_points': None, 'moments': None}}
            # raise ValueError(f'Number of points in 3D must be at least 4. Number of input points:
            # {points_df.shape[0]}.')
        elif n_dimensions == 4 and n_points <= 4:
            logger.warning(f'{points_key}: Number of points in 4D must be at least 5. Number of input points: '
                           f'{points_df.shape[0]}.')
            return {'encoding_failed': {'dist': None, 'ref_points': None, 'moments': None}}
            # raise ValueError(f'Number of points in 4D must be at least 5. Number of input points:
            # {points_df.shape[0]}.')
        elif n_dimensions == 6 and n_points <= 6:
            logger.warning(f'{points_key}: Number of points in 6D must be at least 7. Number of input points: '
                           f'{points_df.shape[0]}.')
            return {'encoding_failed': {'dist': None, 'ref_points': None, 'moments': None}}
            # raise ValueError(f'Number of points in 6D must be at least 7. Number of input points:
            # {points_df.shape[0]}.')
        elif n_dimensions > 6:
            logger.warning(f'{points_key}: Unexpected points dimension: {points_df.shape[1]}. Not implemented.')
            # raise ValueError(f'Unexpected points dimension: {points_df.shape[1]}. Not implemented.')

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
           - ref1, centroid
           - ref2, closest atom to ref1
           - ref3, farthest atom to ref1
           - ref4, farthest atom to ref3
        2. Calculate distances (distance distribution) from reference points to all other points.
        3. Calculate first, second, and third moment for each distance distribution.

        References
        ----------
        [1]_ Ballester and Richards, "Ultrafast shape Recognition to search compound databases for similar molecular
        shapes", J Comput Chem, 2007.
        """

        if points.shape[1] != 3:
            raise ValueError(f'Dimension of input (points) must be 3 but is {points.shape[1]}.')
        if points.shape[0] < 4:
            raise ValueError(f'Number of points must be at least 4 but is {points.shape[0]}.')

        # Get centroid of input coordinates
        ref1 = points.mean(axis=0)

        # Get distances from ref1 to all other points
        dist_ref1 = self._calc_distances_to_point(points, ref1)

        # Get closest and farthest atom to ref1
        ref2, ref3 = points.loc[dist_ref1.idxmin()], points.loc[dist_ref1.idxmax()]

        # Get distances from ref2 to all other points, get distances from ref3 to all other points
        dist_ref2 = self._calc_distances_to_point(points, ref2)
        dist_ref3 = self._calc_distances_to_point(points, ref3)

        # Get farthest atom to ref3
        ref4 = points.loc[dist_ref3.idxmax()]

        # Get distances from ref4 to all other points
        dist_ref4 = self._calc_distances_to_point(points, ref4)

        reference_points = [ref1, ref2, ref3, ref4]
        distances = [dist_ref1, dist_ref2, dist_ref3, dist_ref4]

        return self._get_shape_dict(reference_points, distances)

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
           - ref1, centroid
           - ref2, farthest atom to ref1
           - ref3, farthest atom to ref2
           - ref4, cross product of two vectors spanning ref1, ref2, and ref3
        2. Calculate distances (distance distribution) from reference points to all other points.
        3. Calculate first, second, and third moment for each distance distribution.

        References
        ----------
        [1]_ Armstrong et al., "Molecular similarity including chirality", J Mol Graph Mod, 2009
        """

        if points.shape[1] != 3:
            raise ValueError(f'Dimension of input (points) must be 3 but is {points.shape[1]}.')
        if points.shape[0] < 4:
            raise ValueError(f'Number of points must be at least 4 but is {points.shape[0]}.')

        # Get centroid of input coordinates, and distances from ref1 to all other points
        ref1 = points.mean(axis=0)
        dist_ref1 = self._calc_distances_to_point(points, ref1)

        # Get farthest atom to ref1, and distances from ref2 to all other points
        ref2 = points.loc[dist_ref1.idxmax()]
        dist_ref2 = self._calc_distances_to_point(points, ref2)

        # Get farthest atom to ref2, and distances from ref3 to all other points
        ref3 = points.loc[dist_ref2.idxmax()]
        dist_ref3 = self._calc_distances_to_point(points, ref3)

        # Get forth reference point, including chirality information
        ref4 = self._calc_scaled_3d_cross_product(ref1, ref2, ref3, 'half_norm_a')

        # Get distances from ref4 to all other points
        dist_ref4 = self._calc_distances_to_point(points, ref4)

        reference_points = [ref1, ref2, ref3, ref4]
        distances = [dist_ref1, dist_ref2, dist_ref3, dist_ref4]

        return self._get_shape_dict(reference_points, distances)

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
           - ref1, centroid
           - ref2, farthest atom to ref1
           - ref3, farthest atom to ref2
           - ref4 and ref5, cross product of 2 vectors spanning ref1, ref2, and ref3; 4th dimensions set to maximum and
             minimum value in 4th dimension of input points
        2. Calculate distances (distance distribution) from reference points to all other points.
        3. Calculate first, second, and third moment for each distance distribution.

        References
        ----------
        [1]_ Armstrong et al., "ElectroShape: fast molecular similarity calculations incorporating shape, chirality and
        electrostatics", J Comput Aided Mol Des, 2010.
        """

        if points.shape[1] != 4:
            raise ValueError(f'Dimension of input (points) must be 4 but is {points.shape[1]}.')
        if points.shape[0] < 5:
            raise ValueError(f'Number of points must be at least 5 but is {points.shape[0]}.')

        # Get centroid of input coordinates (in 4 dimensions), and distances from ref1 to all other points
        ref1 = points.mean(axis=0)
        dist_ref1 = self._calc_distances_to_point(points, ref1)

        # Get farthest atom to ref1 (in 4 dimensions), and distances from ref2 to all other points
        ref2 = points.loc[dist_ref1.idxmax()]
        dist_ref2 = self._calc_distances_to_point(points, ref2)

        # Get farthest atom to ref2 (in 4 dimensions), and distances from ref3 to all other points
        ref3 = points.loc[dist_ref2.idxmax()]
        dist_ref3 = self._calc_distances_to_point(points, ref3)

        # Get forth and fifth reference point:
        # a) Get first three dimensions
        c_s = self._calc_scaled_3d_cross_product(ref1, ref2, ref3, 'half_norm_a')

        # b) Add forth dimension with maximum and minmum of points' 4th dimension
        max_value_4thdim = max(points.iloc[:, [3]].values)[0]
        min_value_4thdim = min(points.iloc[:, [3]].values)[0]
        ref4 = c_s.append(pd.Series([scaling_factor * max_value_4thdim], index=[points.columns[3]]))
        ref5 = c_s.append(pd.Series([scaling_factor * min_value_4thdim], index=[points.columns[3]]))

        # Get distances from ref4 and ref5 to all other points
        dist_ref4 = self._calc_distances_to_point(points, ref4)
        dist_ref5 = self._calc_distances_to_point(points, ref5)

        reference_points = [ref1, ref2, ref3, ref4, ref5]
        distances = [dist_ref1, dist_ref2, dist_ref3, dist_ref4, dist_ref5]

        return self._get_shape_dict(reference_points, distances)

    def _calc_shape_6dim_ratar1(self, points, scaling_factor=1):
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
           - ref1, centroid
           - ref2, closest atom to ref1
           - ref3, farthest atom to ref1
           - ref4, farthest atom to ref3
           - ref5, nearest atom to translated and scaled cross product of two vectors spanning ref1, ref2, and ref3
           - ref6, nearest atom to translated and scaled cross product of two vectors spanning ref1, ref3, and ref4
           - ref7, nearest atom to translated and scaled cross product of two vectors spanning ref1, ref4, and ref2
        2. Calculate distances (distance distribution) from reference points to all other points.
        3. Calculate first, second, and third moment for each distance distribution.
        """

        if points.shape[1] != 6:
            raise ValueError(f'Dimension of input (points) must be 6 but is {points.shape[1]}.')
        if points.shape[0] < 7:
            raise ValueError(f'Number of points must be at least 7 but is {points.shape[0]}.')

        # Get centroid of input coordinates (in 6 dimensions), and distances from ref1 to all other points
        ref1 = points.mean(axis=0)
        dist_ref1 = self._calc_distances_to_point(points, ref1)

        # Get closest and farthest atom to centroid ref1 (in 6 dimensions),
        # and distances from ref2 and ref3 to all other points
        ref2, ref3 = points.loc[dist_ref1.idxmin()], points.loc[dist_ref1.idxmax()]
        dist_ref2 = self._calc_distances_to_point(points, ref2)
        dist_ref3 = self._calc_distances_to_point(points, ref3)

        # Get farthest atom to farthest atom to centroid ref2 (in 6 dimensions),
        # and distances from ref3 to all other points
        ref4 = points.loc[dist_ref3.idxmax()]
        dist_ref4 = self._calc_distances_to_point(points, ref4)

        # Get scaled cross product
        ref5_3d = self._calc_scaled_3d_cross_product(ref1, ref2, ref3, 'mean_norm')  # FIXME order of importance, right?
        ref6_3d = self._calc_scaled_3d_cross_product(ref1, ref3, ref4, 'mean_norm')
        ref7_3d = self._calc_scaled_3d_cross_product(ref1, ref4, ref2, 'mean_norm')

        # Get remaining reference points as nearest atoms to scaled cross products
        ref5 = self._calc_nearest_point(ref5_3d, points, scaling_factor)
        ref6 = self._calc_nearest_point(ref6_3d, points, scaling_factor)
        ref7 = self._calc_nearest_point(ref7_3d, points, scaling_factor)

        # Get distances from ref5, ref6, and ref7 to all other points
        dist_ref5 = self._calc_distances_to_point(points, ref5)
        dist_ref6 = self._calc_distances_to_point(points, ref6)
        dist_ref7 = self._calc_distances_to_point(points, ref7)

        reference_points = [ref1, ref2, ref3, ref4, ref5, ref6, ref7]
        distances = [dist_ref1, dist_ref2, dist_ref3, dist_ref4, dist_ref5, dist_ref6, dist_ref7]

        return self._get_shape_dict(reference_points, distances)

    @staticmethod
    def _calc_scaled_3d_cross_product(point_origin, point_a, point_b, scaled_by):
        """
        Calculates a translated and scaled 3D cross product vector based on three input vectors.

        Parameters
        ----------
        point_origin : pandas.Series
            Point with a least N dimensions (N > 2).
        point_a : pandas.Series
            Point with a least N dimensions (N > 2).
        point_b : pandas.Series
            Point with a least N dimensions (N > 2).
        scaled_by : str
            Method to scale the cross product vector:
            - 'half_norm_a': scaled by half the norm of point_a (including all dimensions)
            - 'mean_norm': scaled by the mean norm of point_a and point_b (including first 3 dimensions)

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

        if len({point_origin.size, point_a.size, point_b.size}) > 1:
            raise ValueError(f'The three input pandas.Series are not of same length: '
                             f'{[point_origin.size, point_a.size, point_b.size]}')
        if point_origin.size < 3:
            raise ValueError('The three input pandas.Series are not at least of length 3.')

        # Span vectors to point a and b from origin point
        a = point_a - point_origin
        b = point_b - point_origin

        # Calculate cross product
        cross = np.cross(a[0:3], b[0:3])

        # Calculate norm of cross product
        cross_norm = np.linalg.norm(cross)
        if cross_norm == 0:
            raise ValueError(f'Cross product is zero, thus vectors are linear dependent.')

        # Calculate unit vector of cross product
        cross_unit = cross / cross_norm

        # Calculate value to scale cross product vector
        scaled_by_list = 'half_norm_a mean_norm'.split()

        if scaled_by == scaled_by_list[0]:
            # Calculate half the norm of first vector (including all dimensions)
            scaled_scalar = np.linalg.norm(a) / 2

        elif scaled_by == scaled_by_list[1]:
            # Calculate mean of both vector norms (including first 3 dimensions)
            scaled_scalar = (np.linalg.norm(a[0:3]) + np.linalg.norm(b[0:3])) / 2
        else:
            raise ValueError(f'Scaling method unknown: {scaled_by}. Use: {", ".join(scaled_by_list)}')

        # Scale cross product to length of the mean of both vectors described by the cross product
        cross_scaled = cross_unit * scaled_scalar

        # Move scaled cross product so that it originates from origin point
        c_3d = point_origin[0:3] + pd.Series(cross_scaled, index=point_origin[0:3].index)

        return c_3d

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

        # TODO include test for linear independence / degeneracy of reference points

        # Store reference points as DataFrame
        ref_points = pd.concat(ref_points, axis=1).transpose()
        ref_points.index = [f'ref{i+1}' for i, j in enumerate(ref_points.index)]

        # Store distance distributions as DataFrame
        dist = pd.concat(dist, axis=1)
        dist.columns = [f'dist_ref{i+1}' for i, j in enumerate(dist.columns)]

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
            raise ValueError(f'Input point has {point.size} dimensions. Must have 3 dimensions.')
        if not points.shape[1] > 2:
            raise ValueError(f'Input points have {points.size} dimensions. Must have at least 3 dimensions.')

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
        dist : pandas.DataFrame
            Distance distribution, i.e. distances from reference point to all representatives (points)

        Returns
        -------
        pandas.DataFrame
            First, second, and third moment of distance distribution.
        """

        # Get first, second, and third moment (mean, standard deviation, and skewness) for a distance distribution
        # Second and third moment: delta degrees of freedom = 0 (divisor N)
        if len(dist) > 0:
            m1 = dist.mean()
            m2 = dist.std(ddof=0)
            m3 = pd.Series(cbrt(moment(dist, moment=3)), index=dist.columns.tolist())
        else:
            # In case there is only one data point.
            # However, this should not be possible due to restrictions in get_shape function.
            logger.info(f'Only one data point available for moment calculation, thus write None to moments.')
            m1, m2, m3 = None, None, None

        # Store all moments in DataFrame
        moments = pd.concat([m1, m2, m3], axis=1)
        moments.columns = ['m1', 'm2', 'm3']

        return moments

    @staticmethod
    def _reorder_nested_dict_keys(nested_dict, key_order):
        """
        Change the key order of the nested dictionary data_pseudocenter_subsets (Shapes attribute).
        Example: 'pc/z123/H/6Dratar1/moments' is changed to 'pc/z123/6Dratar1/H/moments'.

        Returns
        -------
        dict of pandas.DataFrames
            Dictionary of DataFrames.
        """

        flat_dict = flatten(nested_dict, reducer='path')

        keys_old = flat_dict.keys()

        if len(set([len(i.split('/')) for i in keys_old])) > 1:
            raise KeyError(f'Flattened keys are nested differently: {[len(i.split("/")) for i in keys_old]}')
        elif [len(i.split('/')) for i in keys_old][0] != len(key_order):
            raise ValueError(f'Key order length ({len(key_order)}) does not match nested levels in dictionary ({len(keys_old)}).')

        keys_new = [i.split('/') for i in keys_old]
        keys_new = [[i[j] for j in key_order] for i in keys_new]
        keys_new = ['/'.join(i) for i in keys_new]

        for key_old, key_new in zip(keys_old, keys_new):
            flat_dict[key_new] = flat_dict.pop(key_old)

        return flat_dict


def process_encoding(molecule_path, output_dir, remove_solvent=False):
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
    molecule_path : str
        Path to molecule structure file(s), can include a wildcard to match multiple files.
    output_dir : str
        Output directory.
    remove_solvent : bool
        Solvent atoms are removed when set to True (default: False).

    Examples
    --------
    >>> from ratar.encoding import process_encoding
    >>> molecule_path = Path(__name__).parent / 'ratar' / 'tests' / 'data' / 'AAK1_4wsq_altA_chainA.mol2'
    >>> output_dir = Path(__name__).parent / 'ratar' / 'tests' / 'data' / 'tmp'
    >>> process_encoding(molecule_path, output_dir)
    """

    # Get all molecule structure files
    molecule_path_list = glob.glob(str(molecule_path))  # Path.glob(<pattern>) not as convenient, glob.glob needs str

    if len(molecule_path_list) == 0:
        logger.info(f'Input path matches no molecule files: {molecule_path}', extra={'molecule_id': 'all'})
    else:
        logger.info(f'Input path matches {len(molecule_path_list)} molecule file(s): {molecule_path}',
                    extra={'molecule_id': 'all'})

    # Get number of molecule structure files and set molecule structure counter
    mol_sum = len(molecule_path_list)

    # Iterate over all binding sites (molecule structure files)
    for mol_counter, mol_path in enumerate(molecule_path_list, 1):

        # Load binding site from molecule structure file
        molecule_loader = MoleculeLoader(mol_path, remove_solvent)

        # Get number of molecule objects and set molecule counter
        molecule_sum = len(molecule_loader.molecules)

        # Iterate over all binding sites in molecule structure file
        for molecule_counter, molecule in enumerate(molecule_loader.molecules, 1):
            print(molecule_counter)

            # Get iteration progress
            logger.info(f'Encoding: {mol_counter}/{mol_sum} molecule structure file - '
                        f'{molecule_counter}/{molecule_sum} molecule',
                        extra={'molecule_id': molecule.code})

            # Process single binding site:

            # Create output folder
            molecule_id_encoding = Path(output_dir) / 'encoding' / molecule.code
            molecule_id_encoding.mkdir(parents=True, exist_ok=True)

            # Encode binding site
            binding_site = BindingSite(molecule)

            # Save binding site
            save_binding_site(binding_site, molecule_id_encoding / 'ratar_encoding.p')

            # Save binding site reference points as cgo file
            save_cgo_file(binding_site, molecule_id_encoding / 'ref_points_cgo.py')


def save_binding_site(binding_site, output_path):
    """
    Save an encoded binding site to a pickle file in an output directory.

    Parameters
    ----------
    binding_site : encoding.BindingSite
        Encoded binding site.
    output_path : str
        Path to output file.

    Examples
    --------
    >>> from ratar.auxiliary import MoleculeLoader
    >>> from ratar.encoding import BindingSite, save_binding_site

     >>> molecule_path = Path(__name__).parent / 'ratar' / 'tests' / 'data' / 'AAK1_4wsq_altA_chainA.mol2'

    >>> molecule_loader = MoleculeLoader(molecule_path, remove_solvent=True)
    >>> molecule = molecule_loader.molecules[0]

    >>> binding_site = BindingSite(molecule)
    >>> save_binding_site(binding_site, Path(__name__).parent / 'ratar' / 'tests' / 'data' / 'tmp' / 'bindingsite.p')
)
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

    Examples
    --------
    >>> from ratar.auxiliary import MoleculeLoader
    >>> from ratar.encoding import BindingSite, save_cgo_file

     >>> molecule_path = Path(__name__).parent / 'ratar' / 'tests' / 'data' / 'AAK1_4wsq_altA_chainA.mol2'

    >>> molecule_loader = MoleculeLoader(molecule_path, remove_solvent=True)
    >>> molecule = molecule_loader.molecules[0]

    >>> binding_site = BindingSite(molecule)
    >>> save_cgo_file(binding_site, '/path/to/output/directory')
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
            obj_name = f'{key.replace("/", "_").replace("_ref_points", "")}__{binding_site.molecule.code}'
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
    lines.append(f'cmd.group("{binding_site.molecule.code[:4]}_ref_points", "{" ".join(obj_names)}")')

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

