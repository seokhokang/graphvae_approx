import numpy as np
import pickle as pkl
from sklearn.preprocessing import StandardScaler
from GVAE import Model
import sys


data = sys.argv[1]

if data=='QM9':
    atom_list=['C','N','O','F']
    
elif data=='ZINC':
    atom_list=['C','N','O','F','P','S','Cl','Br','I']

elif data=='CHEMBL':
    atom_list=['B','C','N','O','F','Si','P','S','Cl','Se','Br','I']
    
data_path = './'+data+'_graph.pkl'
save_dict = './'

print(':: load data')
with open(data_path,'rb') as f:
    [DV, DE, DY, Dsmi] = pkl.load(f)

DV = DV.todense()
DE = DE.todense()
DY = DY

n_node = DV.shape[1]
dim_node = DV.shape[2]
dim_edge = DE.shape[3]
dim_y = DY.shape[1]

print(':: preprocess data')
scaler = StandardScaler()
scaler.fit(DY)
DY = scaler.transform(DY)

dim_atom = len(atom_list)
edge_clip = np.max(DE, (0,1,2))[3:]
mu_prior=np.mean(DY,0)   
cov_prior=np.cov(DY.T)             

model = Model(n_node, dim_node, dim_edge, dim_y, dim_atom, edge_clip, mu_prior, cov_prior)

print('edge_clip', edge_clip)
print(':: train model')
with model.sess:
    load_path=None
    save_path=save_dict+data+'_model.ckpt'
    model.train(DV, DE, DY, Dsmi, atom_list, load_path, save_path)