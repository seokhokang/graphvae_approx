import numpy as np
import pickle as pkl
from sklearn.preprocessing import StandardScaler
from GVAE import Model
import sys


data = sys.argv[1]

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

n_node = n_max

data_path = './'+data+'_graph.pkl'
save_dict = './'

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

print(':: train model')
with model.sess:
    save_path=save_dict+data+'_model.ckpt'
    model.train(DV, DE, DY, Dsmi, atom_list, save_path)