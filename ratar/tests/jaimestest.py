
def test_behaviour_DS_style(a, b):
    assert OldImplementation(a,b) == NewImplementation(a,b)


@pytest.mark.parameterize("a, b, r", [
    (2, 2, 4),
    (3, 5, 8)
])
def test_sum_op(a, b, r)
    summer = SumOp(a,b)
    assert summer.run() == r


##### ENCODING

def test_parser(filepath):
    mol = Molecule(filepath)
    assert mol.n_atoms == <number of atoms>
    assert mol.centroid() == <centroid>

def test_representatives(filepath):
    mol = Molecule(filepath)
    representatives = Representatives.from_molecule(mol)
    assert len(representatives.get_ca()) == 12
    assert len(representatives.get_pca()) == 10
    assert len(representatives.get_pc()) == 54
    assert representatives.get_pc()[0].name == 'CA1'

def test_physchemprop(filepath):
    pcp = Physchemprop()