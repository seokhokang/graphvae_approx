import numpy as np
import pickle as pkl
import sys, sparse
from util import atomFeatures, bondFeatures, _vec_to_mol
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

data = sys.argv[1]
    
if data=='QM9':
    data_size=100000
    n_max=9
    atom_list=['C','N','O','F']

elif data=='ZINC':
    data_size=100000
    n_max=38
    atom_list=['C','N','O','F','P','S','Cl','Br','I']

elif data=='CHEMBL':
    data_size=100000
    n_max=86
    atom_list=['B','C','N','O','F','Si','P','S','Cl','Se','Br','I']

bond_list=['SINGLE', 'DOUBLE', 'TRIPLE']

smisuppl = pkl.load(open('./'+data+'_smi.pkl','rb'))

bpatt1 = Chem.MolFromSmarts('*-[#6;D2]-[#6;D2]-*')#3
bpatt2 = Chem.MolFromSmarts('*-[#6;D2]-*')#3
bpatt3 = Chem.MolFromSmarts('*-[#6;D2]=[#6;D2]-*')#2
bpatt4 = Chem.MolFromSmarts('*=[#6;D2]-[#6;D2]=*')#1
bpatt5 = Chem.MolFromSmarts('*-[#7;D2]-*')#1
bpatt6 = Chem.MolFromSmarts('*-[#8;D2]-*')#1

bpatt_list = [bpatt1, bpatt2, bpatt3, bpatt4, bpatt5, bpatt6]
bpatt_dim = [3, 3, 2, 1, 1, 1]

dim_node = 5 + len(atom_list)
dim_edge = len(bond_list) + len(bpatt_list)

current_max = 0

DV = []
DE = [] 
DY = []
Dsmi = []
for i, smi in enumerate(smisuppl):

    smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi),isomericSmiles=False)
    mol = Chem.MolFromSmiles(smi)  
    
    Chem.Kekulize(mol)
    n_atom = mol.GetNumHeavyAtoms()

    # node DV
    node = np.zeros((n_atom, dim_node), dtype=np.int8)
    for j in range(n_atom):
        atom = mol.GetAtomWithIdx(j)
        node[j, :]=atomFeatures(atom, atom_list)

    # edge pattern search
    del_ids = []
    case_list = []
    for j in range(len(bpatt_list)):
        for aids in mol.GetSubstructMatches(bpatt_list[j]):
            if np.sum([(aid in del_ids) for aid in aids]) == 0 and np.sum(np.abs([mol.GetAtomWithIdx(aid).GetFormalCharge() for aid in aids])) == 0:
                del_ids = del_ids + list(aids[1:-1])
                case_list.append([j, np.min([aids[0], aids[-1]]), np.max([aids[0], aids[-1]])])

    # edge DE
    edge = np.zeros((n_atom, n_atom, dim_edge), dtype=np.int8)
    for j in range(n_atom - 1):
        for k in range(j + 1, n_atom):
            molpath = Chem.GetShortestPath(mol, j, k)
            bonds = [mol.GetBondBetweenAtoms(molpath[bid], molpath[bid + 1]) for bid in range(len(molpath) - 1)]
            edge[j, k, :] = np.concatenate([bondFeatures(bonds, bond_list), np.zeros(len(bpatt_list))], 0)
            for m in range(len(bpatt_list)):
                if [m, j, k] in case_list:
                    assert case_list.count([m, j, k]) <= bpatt_dim[m]
                    edge[j, k, len(bond_list) + m] = case_list.count([m, j, k])

            edge[k, j, :] = edge[j, k, :]
    
    # compression
    node = np.delete(node, del_ids, 0)
      
    edge = np.delete(edge, del_ids, 0)
    edge = np.delete(edge, del_ids, 1)
    
    node = np.pad(node, ((0, n_max - node.shape[0]),(0, 0)))
    edge = np.pad(edge, ((0, n_max - edge.shape[0]), (0, n_max - edge.shape[1]),(0, 0))) 

    # property DY
    property = [Descriptors.ExactMolWt(mol), Descriptors.MolLogP(mol)]
        
    assert smi == _vec_to_mol(node, edge, atom_list, bpatt_dim):
    
    if current_max < node.shape[0]:
        current_max = node.shape[0]
        print('current max = ', n_atom, current_max, smi)

    # append
    DV.append(node)
    DE.append(edge)
    DY.append(property)
    Dsmi.append(smi)

    if i % 1000 == 0:
        print(i, flush=True)

    if len(DV) == data_size: break

# np array    
DV = np.asarray(DV, dtype=np.int8)
DE = np.asarray(DE, dtype=np.int8)
DY = np.asarray(DY)
Dsmi = np.asarray(Dsmi)

DV = DV[:,:current_max,:]
DE = DE[:,:current_max,:current_max,:]

# compression
DV = sparse.COO.from_numpy(DV)
DE = sparse.COO.from_numpy(DE)

print(DV.shape, DE.shape, DY.shape)

# save
with open(data+'_graph.pkl','wb') as fw:
    pkl.dump([DV, DE, DY, Dsmi], fw)