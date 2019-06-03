from ratar.auxiliary import MolFileLoader
from ratar.encoding import BindingSite

path1 = "/home/dominique/Documents/data/klifs/test_20190506/structures/HUMAN/AAK1/4wsq_altA_chainA/pocket.mol2"
path2 = "/home/dominique/Documents/data/klifs/test_20190506/structures/HUMAN/AAK1/4wsq_altA_chainB/pocket.mol2"

pmols1 = MolFileLoader(path1)
pmols2 = MolFileLoader(path2)

bs1 = BindingSite(pmols1.pmols[0])
bs2 = BindingSite(pmols1.pmols[0])
bs3 = BindingSite(pmols2.pmols[0])

print(bs1 == bs2)
print(bs1 == bs3)