import numpy as np
import pickle as pkl
import sys, sparse
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.preprocessing import StandardScaler
from GVAE import Model


data = 'QM9'

if data=='QM9':

    data_size=100000
    n_max=9
    dim_node=2 + 3 + 4
    dim_edge=3
    
    atom_list=['C','N','O','F']
    target_list=[[120,125,130],[-0.4,0.2,0.8]]

elif data=='ZINC':

    data_size=100000
    n_max=38
    dim_node=2 + 3 + 9
    dim_edge=3
    
    atom_list=['C','N','O','F','P','S','Cl','Br','I']
    target_list=[[300,350,400],[1.5,2.5,3.5]]

n_node = n_max

data_path = './'+data+'_graph.pkl'
save_path = './'+data+'_model.ckpt'

print(':: load data')
with open(data_path,'rb') as f:
    [DV, DE, DY, Dsmi] = pkl.load(f)

DV = DV.todense()
DE = DE.todense()
DY = DY

dim_y = DY.shape[1]

print(':: preprocess data')
scaler = StandardScaler()
scaler.fit(DY)
DY = scaler.transform(DY)

mu_prior=np.mean(DY,0)   
cov_prior=np.cov(DY.T)             

model = Model(n_node, dim_node, dim_edge, dim_y, mu_prior, cov_prior)
np.set_printoptions(precision=3, suppress=True)

with model.sess:
    model.saver.restore(model.sess, save_path)     

    # unconditional generation     
    total_count, valid_count, novel_count, unique_count, genmols = model.test(10000, 0, Dsmi, atom_list)

    valid=valid_count/total_count
    unique=unique_count/valid_count
    novel=novel_count/valid_count
    
    print(':: Valid:',valid*100,'Unique:',unique*100,'Novel:',novel*100,'GMean:', 100*(valid*unique*novel)**(1/3))
    
    list_Y=[]
    for m in genmols:
        mol = Chem.MolFromSmiles(m)
        list_Y.append([Descriptors.ExactMolWt(mol), Descriptors.MolLogP(mol)])

    print(':: unconditional generation results', len(genmols), np.mean(list_Y,0), np.std(list_Y,0))

    # conditional generation 
    for target_id in range(len(target_list)):
        for target_Y in target_list[target_id]:
        
            target_Y_norm=(target_Y-scaler.mean_[target_id])/(scaler.var_[target_id]**0.5)
        
            total_count, valid_count, novel_count, unique_count, genmols = model.test(10000, 1, Dsmi, atom_list, target_id, target_Y_norm)
        
            valid=valid_count/total_count
            unique=unique_count/valid_count
            novel=novel_count/valid_count
        
            print(':: Valid:',valid*100,'Unique:',unique*100,'Novel:',novel*100,'GMean:', 100*(valid*unique*novel)**(1/3))
            
            list_Y=[]
            for i, m in enumerate(genmols):
                mol = Chem.MolFromSmiles(m)
                list_Y.append([Descriptors.ExactMolWt(mol), Descriptors.MolLogP(mol)])
            
            print(':: conditional generation results', target_id, target_Y, len(genmols), np.mean(list_Y,0), np.std(list_Y,0))