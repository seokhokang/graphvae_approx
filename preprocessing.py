import numpy as np
import pickle as pkl
import sys, sparse
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors


def to_onehot(val, cat):

    vec = np.zeros(len(cat))
    for i, c in enumerate(cat):
        if val == c: vec[i] = 1

    if np.sum(vec) == 0: print('* exception: missing category', val)
    assert np.sum(vec) == 1

    return vec

def atomFeatures(a):

    v1 = to_onehot(a.GetFormalCharge(), [-1, 1, 0])[:2]
    v2 = to_onehot(a.GetNumExplicitHs(), [1, 2, 3, 0])[:3]    
    v3 = to_onehot(a.GetSymbol(), atom_list)
    
    return np.concatenate([v1, v2, v3], axis=0)

def bondFeatures(bonds):

    e1 = np.zeros(3)
    if len(bonds)==1:
        e1 = to_onehot(str(bonds[0].GetBondType()), ['SINGLE', 'DOUBLE', 'TRIPLE'])

    return np.array(e1)


data = 'QM9'

if data=='QM9':

    data_size=100000
    n_max=9
    dim_node=2 + 3 + 4
    dim_edge=3
    
    atom_list=['C','N','O','F']

elif data=='ZINC':

    data_size=100000
    n_max=38
    dim_node=2 + 3 + 9
    dim_edge=3
    
    atom_list=['C','N','O','F','P','S','Cl','Br','I']
    
smisuppl = pkl.load(open('./'+data+'_smi.pkl','rb'))
molsuppl = np.array([Chem.MolFromSmiles(smi) for smi in smisuppl])

DV = []
DE = [] 
DY = []
Dsmi = []
for i, mol in enumerate(molsuppl):

    mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol,isomericSmiles=False))  
    Chem.Kekulize(mol)
    n_atom = mol.GetNumHeavyAtoms()
        
    # node DV
    node = np.zeros((n_max, dim_node), dtype=int)
    for j in range(n_atom):
        atom = mol.GetAtomWithIdx(j)
        node[j, :]=atomFeatures(atom)
    
    # edge DE
    edge = np.zeros((n_max, n_max, dim_edge), dtype=int)
    for j in range(n_atom - 1):
        for k in range(j + 1, n_atom):
            molpath = Chem.GetShortestPath(mol, j, k)
            bonds = [mol.GetBondBetweenAtoms(molpath[bid], molpath[bid + 1]) for bid in range(len(molpath) - 1)]
            edge[j, k, :] = bondFeatures(bonds)
            edge[k, j, :] = edge[j, k, :]

    # property DY
    property = [Descriptors.ExactMolWt(mol), Descriptors.MolLogP(mol)]

    # append
    DV.append(node)
    DE.append(edge)
    DY.append(property)
    Dsmi.append(Chem.MolToSmiles(mol))

    if i % 1000 == 0:
        print(i, flush=True)

    if len(DV) == data_size: break

# np array    
DV = np.asarray(DV, dtype=int)
DE = np.asarray(DE, dtype=int)
DY = np.asarray(DY)
Dsmi = np.asarray(Dsmi)

# compression
DV = sparse.COO.from_numpy(DV)
DE = sparse.COO.from_numpy(DE)

# save
with open(data+'_graph.pkl','wb') as fw:
    pkl.dump([DV, DE, DY, Dsmi], fw)