import numpy as np
from rdkit import Chem

def to_onehot(val, cat):

    vec = np.zeros(len(cat))
    for i, c in enumerate(cat):
        if val == c: vec[i] = 1

    if np.sum(vec) == 0: print('* exception: missing category', val)
    assert np.sum(vec) == 1

    return vec

def atomFeatures(atom, atom_list):

    v1 = to_onehot(atom.GetFormalCharge(), [-1, 1, 0])[:2]
    v2 = to_onehot(atom.GetNumExplicitHs(), [1, 2, 3, 0])[:3]    
    v3 = to_onehot(atom.GetSymbol(), atom_list)
    
    return np.concatenate([v1, v2, v3], axis=0)

def bondFeatures(bonds, bond_list):

    e1 = np.zeros(len(bond_list))
    if len(bonds)==1:
        e1 = to_onehot(str(bonds[0].GetBondType()), bond_list)

    return e1
    
def _vec_to_mol(dv, de, atom_list, edge_clip, train=False):
    
    def to_dummy(vec, ax=1, thr=1):  return np.concatenate([vec, thr - np.sum(vec, ax, keepdims=True)], ax)
  
    def to_val(vec, cat):  
        out = np.zeros(np.shape(vec))
        for i, v in enumerate(vec):
            for j, c in enumerate(cat): 
                if v == j: out[i]=c
                
        return out

    bond_ref = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE]
    
    node_atom = np.argmax(to_dummy(dv[:,5:len(atom_list)+5], 1), 1)
    node_charge = to_val(np.argmax(to_dummy(dv[:,:2], 1), 1), [-1, 1])
    node_exp = to_val(np.argmax(to_dummy(dv[:,2:5], 1), 1), [1, 2, 3])  

    edge_bond = np.argmax(to_dummy(de[:,:,:len(bond_ref)], 2), 2)
    edge_patt = np.round(np.clip(de[:,:,len(bond_ref):], [0] * len(edge_clip), edge_clip)).astype(int)

    selid = np.where(node_atom<len(atom_list))[0]
    
    node_atom = node_atom[selid]
    node_charge = node_charge[selid]
    node_exp = node_exp[selid]

    edge_bond = edge_bond[selid][:,selid]
    edge_patt = edge_patt[selid][:,selid]
    
    edmol = Chem.EditableMol(Chem.MolFromSmiles(''))
    
    m = len(node_atom)
    for j in range(m):
        atom_add = Chem.Atom(atom_list[node_atom[j]])
        if node_charge[j] != 0: atom_add.SetFormalCharge(int(node_charge[j]))
        if node_exp[j] > 0: atom_add.SetNumExplicitHs(int(node_exp[j]))
        edmol.AddAtom(atom_add)

    for j in range(m-1):
        for k in range(j+1, m):
            if edge_bond[j, k] < len(bond_ref):
                edmol.AddBond(j, k, bond_ref[edge_bond[j, k]])

    for j in range(len(node_atom)):
  
        for k in range(j + 1, len(node_atom)):
            
            for _ in range(edge_patt[j,k,0]):
                edmol.AddAtom(Chem.Atom('C'))
                edmol.AddAtom(Chem.Atom('C'))
                edmol.AddBond(j, m, bond_ref[0])
                edmol.AddBond(m, m+1, bond_ref[0])
                edmol.AddBond(m+1, k, bond_ref[0])
                m += 2
            
            for _ in range(edge_patt[j,k,1]):
                edmol.AddAtom(Chem.Atom('C'))
                edmol.AddBond(j, m, bond_ref[0])
                edmol.AddBond(m, k, bond_ref[0])
                m += 1
  
            for _ in range(edge_patt[j,k,2]):
                edmol.AddAtom(Chem.Atom('C'))
                edmol.AddAtom(Chem.Atom('C'))
                edmol.AddBond(j, m, bond_ref[0])
                edmol.AddBond(m, m+1, bond_ref[1])
                edmol.AddBond(m+1, k, bond_ref[0])
                m += 2  
                      
            for _ in range(edge_patt[j,k,3]):
                edmol.AddAtom(Chem.Atom('C'))
                edmol.AddAtom(Chem.Atom('C'))
                edmol.AddBond(j, m, bond_ref[1])
                edmol.AddBond(m, m+1, bond_ref[0])
                edmol.AddBond(m+1, k, bond_ref[1])
                m += 2       
                
            for _ in range(edge_patt[j,k,4]):
                edmol.AddAtom(Chem.Atom('N'))
                edmol.AddBond(j, m, bond_ref[0])
                edmol.AddBond(m, k, bond_ref[0])
                m += 1  
                                                      
            for _ in range(edge_patt[j,k,5]):
                edmol.AddAtom(Chem.Atom('O'))
                edmol.AddBond(j, m, bond_ref[0])
                edmol.AddBond(m, k, bond_ref[0])
                m += 1
        
    mol_rec = edmol.GetMol()
    # sanity check
    Chem.SanitizeMol(mol_rec)
        
    mol_n = Chem.MolFromSmiles(Chem.MolToSmiles(mol_rec))
    output = Chem.MolToSmiles(mol_n)  
    
    if train and '.' in output: raise
    
    return output 

def _permutation(set):

    permid = np.random.permutation(len(set[0]))
    for i in range(len(set)):
        set[i] = set[i][permid]

    return set