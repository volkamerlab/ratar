class Representatives:
    """
    Class used to store binding site representatives. Representatives are selected atoms in a binding site,
    for instances all Calpha atoms of a binding site could serve as its representatives.

    Parameters
    ----------
    mol : pandas.DataFrame
        DataFrame containing atom lines of mol2 or pdb file.

    Attributes
    ----------
    repres_dict : dict
        Representatives stored as dictionary with several representation methods serving as key.
        Example: {'ca': ..., 'pca': ..., 'pc': ...}
    """

    def __init__(self, ca=None, pca=None, pc=None):
        self._data = {
            'ca': ca,
            'pca': pca,
            'pc': pc,
        }

    @classmethod
    def from_molecule(cls, molecule):
        data = {
            'ca': None,
            'pca': None,
            'pc': None,
        }
        for key in data.keys():
            data[key] = _get_representatives(molecule, key)
        cls.__init__(**data)

    @property
    def ca(self):
        return self._data['ca']

    @ca.setter
    def ca(self, value):
        # Make tests on value
        self._data['ca'] = value

    @property
    def pca(self):
        return self._data['pca']

    @property
    def pc(self):
        return self._data['pc']