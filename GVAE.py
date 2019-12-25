import numpy as np
import tensorflow as tf
import sys, time, warnings
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors

class Model(object):

    def __init__(self, n_node, dim_node, dim_edge, dim_y, mu_prior, cov_prior, dim_h=50, dim_z=100, dim_f=500, n_mpnn_step=3, n_dummy=5, batch_size=20, lr=0.0005, useGPU=True, use_PREFERENCE=False):
        
        warnings.filterwarnings('ignore')
        tf.logging.set_verbosity(tf.logging.ERROR)
        rdBase.DisableLog('rdApp.error') 
        rdBase.DisableLog('rdApp.warning')
 
        if use_PREFERENCE: self.dim_R = 2
        else: self.dim_R = 1
        
        self.n_node=n_node
        self.dim_node=dim_node
        self.dim_edge=dim_edge
        self.dim_y=dim_y
        
        self.mu_prior=mu_prior
        self.cov_prior=cov_prior

        self.dim_h=dim_h
        self.dim_z=dim_z
        self.dim_f=dim_f
        self.n_mpnn_step=n_mpnn_step
        self.n_dummy=n_dummy
        self.batch_size=batch_size
        self.lr=lr

        # variables
        self.G = tf.Graph()
        self.G.as_default()

        self.node = tf.placeholder(tf.float32, [self.batch_size, self.n_node, self.dim_node])
        self.edge = tf.placeholder(tf.float32, [self.batch_size, self.n_node, self.n_node, self.dim_edge])
        self.property = tf.placeholder(tf.float32, [self.batch_size, self.dim_y])

        self.latent = self._encoder(self.batch_size, self.node, self.edge, self.property, self.n_mpnn_step, self.dim_h, self.dim_h * 2, self.dim_z * 2, name='encoder', reuse=False)    
        self.latent_mu, self.latent_lsgms = tf.split(self.latent, [self.dim_z, self.dim_z], 1)
        
        self.latent_epsilon = tf.random_normal([self.batch_size, self.dim_z], 0., 1.)
        self.latent_sample = tf.add(self.latent_mu, tf.multiply(tf.exp(0.5 * self.latent_lsgms), self.latent_epsilon))
        self.latent_sample2 = tf.concat([self.latent_sample, self.property], 1)
        
        self.rec_node, self.rec_edge = self._generator(self.batch_size, self.latent_sample2, self.n_mpnn_step, name='generator', reuse=False)

        self.new_latent = tf.random_normal([self.batch_size, self.dim_z], 0., 1.)
        mngen = tf.contrib.distributions.MultivariateNormalFullCovariance(loc=self.mu_prior, covariance_matrix=self.cov_prior)
        self.new_y = tf.dtypes.cast(mngen.sample(self.batch_size, self.dim_y), tf.float32)
        self.new_latent2 = tf.concat([self.new_latent, self.new_y], 1)
        
        self.new_node, self.new_edge = self._generator(self.batch_size, self.new_latent2, self.n_mpnn_step, name='generator', reuse=True)
        
        self.node_pad = tf.pad(self.node, tf.constant([[0,0],[0,self.n_dummy],[0,0]]), 'CONSTANT')
        self.edge_pad = tf.pad(self.edge, tf.constant([[0,0],[0,self.n_dummy],[0,self.n_dummy],[0,0]]), 'CONSTANT')

        # auxiliary
        self.R_rec = self._encoder(self.batch_size, self.rec_node, self.rec_edge, None, self.n_mpnn_step, self.dim_h, self.dim_h * 2, self.dim_R, name='auxiliary/R', reuse=False)
        self.R_fake = self._encoder(self.batch_size, self.new_node, self.new_edge, None, self.n_mpnn_step, self.dim_h, self.dim_h * 2, self.dim_R, name='auxiliary/R', reuse=True)
        self.R_real = self._encoder(self.batch_size, self.node_pad, self.edge_pad, None, self.n_mpnn_step, self.dim_h, self.dim_h * 2, self.dim_R, name='auxiliary/R', reuse=True)
        
        self.R_rec_t = tf.placeholder(tf.float32, [self.batch_size, self.dim_R])
        self.R_fake_t = tf.placeholder(tf.float32, [self.batch_size, self.dim_R])
        self.R_real_t = tf.placeholder(tf.float32, [self.batch_size, self.dim_R])
        
        self.y_rec = self._encoder(self.batch_size, self.rec_node, self.rec_edge, None, self.n_mpnn_step, self.dim_h, self.dim_h * 2, self.dim_y, name='auxiliary/Y', reuse=False)
        self.y_fake = self._encoder(self.batch_size, self.new_node, self.new_edge, None, self.n_mpnn_step, self.dim_h, self.dim_h * 2, self.dim_y, name='auxiliary/Y', reuse=True)
        self.y_real = self._encoder(self.batch_size, self.node_pad, self.edge_pad, None, self.n_mpnn_step, self.dim_h, self.dim_h * 2, self.dim_y, name='auxiliary/Y', reuse=True)

        # session
        self.saver = tf.train.Saver()
        if useGPU:
            self.sess = tf.Session()
        else:
            config = tf.ConfigProto(device_count = {'GPU': 0} )
            self.sess = tf.Session(config=config)


    def train(self, DV, DE, DY, Dsmi, atom_list, load_path=None, save_path=None):

        def _reward(nodes, edges):
        
            def _preference(smi):
                val = 0
                # mol = Chem.MolFromSmiles(smi)
                # set val = 1 if mol is preferred 
                
                return val
        
            R_t = np.zeros((self.batch_size, self.dim_R))
            for j in range(self.batch_size):
                try:         
                    R_smi = self._vec_to_mol(nodes[j], edges[j], atom_list, train=True)
                    R_t[j, 0] = 1
                    if self.dim_R == 2: R_t[j, 1] = _preference(R_smi)
                except:
                    pass
            
            return R_t

        ## objective function
        cost_KLD = tf.reduce_mean(tf.reduce_sum(self._iso_KLD(self.latent_mu, self.latent_lsgms), 1))
        
        cost_rec1 = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(tf.reduce_sum(self.node, 1), tf.reduce_sum(self.rec_node, 1)), 1))
        cost_rec1 = cost_rec1 + tf.reduce_mean(tf.reduce_sum(tf.squared_difference(tf.reduce_sum(self.edge, [1, 2]), tf.reduce_sum(self.rec_edge, [1, 2])), 1) )

        a = [tf.matmul(self.edge[:,:,:,i], self.node) for i in range(self.dim_edge)]
        ar = [tf.matmul(self.rec_edge[:,:,:,i], self.rec_node) for i in range(self.dim_edge)]
        cost_rec2 = tf.reduce_sum([tf.reduce_mean(tf.reduce_sum(tf.squared_difference(tf.reduce_sum(a[i], 1), tf.reduce_sum(ar[i], 1)), 1)) for i in range(self.dim_edge)])
        
        b = [tf.matmul(tf.transpose(self.node, perm=[0,2,1]), a[i]) for i in range(self.dim_edge)]
        br = [tf.matmul(tf.transpose(self.rec_node, perm=[0,2,1]), ar[i]) for i in range(self.dim_edge)]
        cost_rec3 = tf.reduce_sum([tf.reduce_mean(tf.reduce_sum(tf.squared_difference(b[i], br[i]), [1, 2])) for i in range(self.dim_edge)])
 
        cost_R_VAE = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.R_rec), logits=self.R_rec), 1))
        cost_R_VAE = cost_R_VAE + tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.R_fake), logits=self.R_fake), 1))
        
        cost_R_aux = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.R_real_t, logits=self.R_real), 1))
        cost_R_aux = cost_R_aux + tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.R_fake_t, logits=self.R_fake), 1))
        cost_R_aux = cost_R_aux + tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.R_rec_t, logits=self.R_rec), 1))
        
        cost_Y_VAE = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(self.property, self.y_rec) * self.R_rec_t[:,0:1], 1))
        cost_Y_VAE = cost_Y_VAE + tf.reduce_mean(tf.reduce_sum(tf.squared_difference(self.new_y, self.y_fake) * self.R_fake_t[:,0:1], 1))
        
        cost_Y_aux = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(self.property, self.y_real), 1))

        beta1 = 1
        beta2 = 1
        
        self.cost_VAE = cost_KLD + (cost_rec1 + cost_rec2 + cost_rec3) + beta1 * cost_R_VAE + beta2 * cost_Y_VAE
        self.cost_aux = beta1 * cost_R_aux + beta2 * cost_Y_aux

        ## variable set
        vars_E = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
        vars_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        vars_R = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='auxiliary/R')
        vars_Y = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='auxiliary/Y')
        
        assert len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)) == len(vars_E+vars_G+vars_R+vars_Y)
        
        train_VAE = tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(self.cost_VAE, var_list=vars_E+vars_G)
        train_aux = tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(self.cost_aux, var_list=vars_R+vars_Y)  

        self.sess.run(tf.initializers.global_variables())
        np.set_printoptions(precision=3, suppress=True)

        n_batch = int(len(DV)/self.batch_size)
        
        if load_path is not None:
            self.saver.restore(self.sess, load_path)

        ## tranining
        max_epoch=100
        print('::: training')
        trn_log = np.zeros((max_epoch, 8))
        eval_log = np.zeros(max_epoch)
        for epoch in range(max_epoch):
        
            [DV, DE, DY] = self._permutation([DV, DE, DY])
            
            trnscores = np.zeros((n_batch, 8))	
            for i in range(n_batch):

                start_=i*self.batch_size
                end_=start_+self.batch_size

                assert self.batch_size == end_ - start_
                
                [new_nodes, new_edges, rec_nodes, rec_edges, lat1s, lat2s, lat3s] = self.sess.run([self.new_node, self.new_edge, self.rec_node, self.rec_edge, 
                                                                                                   self.latent_epsilon, self.new_latent, self.new_y],
                                     feed_dict = {self.node: DV[start_:end_], self.edge: DE[start_:end_], self.property: DY[start_:end_]})

                fake_t = _reward(new_nodes, new_edges)
                rec_t = _reward(rec_nodes, rec_edges)
                real_t = _reward(DV[start_:end_], DE[start_:end_])
                                                  
                self.sess.run(train_VAE,
                                     feed_dict = {self.node: DV[start_:end_], self.edge: DE[start_:end_], self.property: DY[start_:end_],
                                                  self.latent_epsilon: lat1s, self.new_latent: lat2s, self.new_y: lat3s, 
                                                  self.R_real_t: real_t, self.R_fake_t: fake_t, self.R_rec_t: rec_t})
                
                trnresult = self.sess.run([train_aux, cost_KLD, cost_rec1, cost_rec2, cost_rec3, cost_R_VAE, cost_R_aux, cost_Y_VAE, cost_Y_aux],
                                     feed_dict = {self.node: DV[start_:end_], self.edge: DE[start_:end_], self.property: DY[start_:end_],
                                                  self.latent_epsilon: lat1s, self.new_latent: lat2s, self.new_y: lat3s, 
                                                  self.R_real_t: real_t, self.R_fake_t: fake_t, self.R_rec_t: rec_t})  
                 
                trnscores[i, :] = trnresult[1:]
            
            trn_log[epoch, :] = np.mean(trnscores, 0)        
            print('--training epoch id: ', epoch, ' trn log: ', trn_log[epoch])  

            if epoch > 0:
                total_count, valid_count, novel_count, unique_count, genmols = self.test(10000, 0, Dsmi, atom_list)
    
                valid_count=valid_count + 1e-7
                valid=valid_count/total_count
                unique=unique_count/valid_count
                novel=novel_count/valid_count
                
                gmean = (valid * unique * novel) ** (1/3)
                eval_log[epoch] = gmean
                
                print('--evaluation epoch id: ', epoch, 'Valid:',valid*100,' // Unique:',unique*100,' // Novel:',novel*100, '// Gmean:',gmean*100)            
          
                if np.max(eval_log[:epoch+1]) == gmean:
                    self.saver.save(self.sess, save_path)              
                            

    def test(self, n_gen, isconditional, smisuppl, atom_list, target_id=None, target_Y_norm=None):

        newsuppl=[]
        
        total_count=0
        valid_count=0
        novel_count=0
        unique_count=0
        for t in range(int(n_gen / self.batch_size)): 

            if isconditional:
                latvecsY = np.concatenate([self._random_cond_normal(target_id, target_Y_norm) for _ in range(self.batch_size)], 0)
                [new_node, new_edge, new_y] = self.sess.run([self.new_node, self.new_edge, self.y_fake], feed_dict = {self.new_y: latvecsY})
            else:
                [new_node, new_edge, new_y] = self.sess.run([self.new_node, self.new_edge, self.y_fake], feed_dict = {})
            
            for i in range(len(new_node)):
                total_count+=1
                try:         
                    smi = self._vec_to_mol(new_node[i], new_edge[i], atom_list, train=False)

                    valid_count+=1
                    
                    if smi not in smisuppl:
                        novel_count+=1
                        
                    if smi not in newsuppl:
                        newsuppl.append(smi)
                        unique_count+=1

                except:
                    pass
        
        return total_count, valid_count, novel_count, unique_count, newsuppl
        

    def _random_cond_normal(self, yid, ytarget):
    
        id2=[yid]
        id1=np.setdiff1d(range(self.dim_y),id2)
    
        mu1=self.mu_prior[id1]
        mu2=self.mu_prior[id2]
        
        cov11=self.cov_prior[id1][:,id1]
        cov12=self.cov_prior[id1][:,id2]
        cov22=self.cov_prior[id2][:,id2]
        cov21=self.cov_prior[id2][:,id1]
        
        cond_mu=np.transpose(mu1.T+np.matmul(cov12, np.linalg.inv(cov22)) * (ytarget-mu2))[0]
        cond_cov=cov11 - np.matmul(np.matmul(cov12, np.linalg.inv(cov22)), cov21)
        
        marginal_sampled=np.random.multivariate_normal(cond_mu, cond_cov, 1)
        
        sample_y=np.zeros(self.dim_y)
        sample_y[id1]=marginal_sampled
        sample_y[id2]=ytarget
        
        return np.asarray([sample_y])
    
    
    def _vec_to_mol(self, dv, de, atom_list, train=True):
    
        def to_dummy(vec, ax=1, thr=1):  return np.concatenate([vec, thr - np.sum(vec, ax, keepdims=True)], ax)

        def to_val(vec, cat):  
            out = np.zeros(np.shape(vec))
            for i, v in enumerate(vec):
                for j, c in enumerate(cat): 
                    if v == j: out[i]=c
                    
            return out
        
        ref_atom = atom_list
        ref_bond = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]
        ref_bond = ref_bond[:self.dim_edge]
    
        node_charge = to_val(np.argmax(to_dummy(dv[:,:2], 1), 1), [-1, 1])
        node_exp = to_val(np.argmax(to_dummy(dv[:,2:2+3], 1), 1), [1, 2, 3])    
    
        node_atom = np.argmax(to_dummy(dv[:,2+3:], 1), 1)
        edge_bond = np.argmax(to_dummy(de, 2), 2)
        
        selid = np.intersect1d(np.where(node_atom<len(ref_atom))[0],np.where(np.min(edge_bond,1)<len(ref_bond))[0])
        if train and len(selid) != len(np.union1d(np.where(node_atom<len(ref_atom))[0],np.where(np.min(edge_bond,1)<len(ref_bond))[0])): raise
            
        node_charge = node_charge[selid]
        node_exp = node_exp[selid]
        node_atom = node_atom[selid]
        edge_bond = edge_bond[selid][:,selid]
    
        edmol = Chem.EditableMol(Chem.MolFromSmiles(''))
        
        m = len(node_atom)
        
        for j in range(m):
            atom_add = Chem.Atom(ref_atom[node_atom[j]])
            if node_charge[j] != 0: atom_add.SetFormalCharge(int(node_charge[j]))
            if node_exp[j] > 0: atom_add.SetNumExplicitHs(int(node_exp[j]))
            edmol.AddAtom(atom_add)
        
        for j in range(m-1):
            for k in range(j+1, m):
                if edge_bond[j, k] < len(ref_bond):
                    edmol.AddBond(j, k, ref_bond[edge_bond[j, k]])
    
        mol_rec = edmol.GetMol()
        # sanity check
        Chem.SanitizeMol(mol_rec)
            
        mol_n = Chem.MolFromSmiles(Chem.MolToSmiles(mol_rec))
        if self.dim_edge == 3:
            Chem.Kekulize(mol_n)
            output = Chem.MolToSmiles(mol_n, kekuleSmiles=True) 
        elif self.dim_edge == 4:
            output = Chem.MolToSmiles(mol_n) 
        else:
            raise
        
        if '.' in output: raise
        
        # additional constraints 
        # rings = mol_n.GetRingInfo().AtomRings()
        #for ring in rings:
        #    if len(ring) > 8:
        #        raise
        
        return output 
    

    def _permutation(self, set):
    
        permid = np.random.permutation(len(set[0]))
        for i in range(len(set)):
            set[i] = set[i][permid]
    
        return set
        
    
    def _encoder(self, batch_size, node, edge, prop, n_step, hiddendim, aggrdim, latentdim, name='', reuse=True):

        def _embed_node(inp):
        
            inp = tf.layers.dense(inp, hiddendim, activation = tf.nn.tanh)

            inp = inp * mask
        
            return inp

        def _edge_nn(inp):

            inp = tf.layers.dense(inp, hiddendim * hiddendim)
        
            inp = tf.reshape(inp, [batch_size, n_node, n_node, hiddendim, hiddendim])
            inp = inp * tf.reshape(1-tf.eye(n_node), [1, n_node, n_node, 1, 1])
            inp = inp * tf.reshape(mask, [batch_size, n_node, 1, 1, 1]) * tf.reshape(mask, [batch_size, 1, n_node, 1, 1])

            return inp

        def _MPNN(edge_wgt, node_hidden, n_step):
        
            def _msg_nn(wgt, node):
            
                wgt = tf.reshape(wgt, [batch_size * n_node, n_node * hiddendim, hiddendim])
                node = tf.reshape(node, [batch_size * n_node, hiddendim, 1])
            
                msg = tf.matmul(wgt, node)
                msg = tf.reshape(msg, [batch_size, n_node, n_node, hiddendim])
                msg = tf.transpose(msg, perm = [0, 2, 3, 1])
                msg = tf.reduce_mean(msg, 3)
            
                return msg

            def _update_GRU(msg, node, reuse_GRU):
            
                with tf.variable_scope('mpnn_gru', reuse=reuse_GRU):
            
                    msg = tf.reshape(msg, [batch_size * n_node, 1, hiddendim])
                    node = tf.reshape(node, [batch_size * n_node, hiddendim])
            
                    cell = tf.nn.rnn_cell.GRUCell(hiddendim)
                    _, node_next = tf.nn.dynamic_rnn(cell, msg, initial_state = node)
            
                    node_next = tf.reshape(node_next, [batch_size, n_node, hiddendim]) * mask
            
                return node_next

            nhs=[]
            for i in range(n_step):
                message_vec = _msg_nn(edge_wgt, node_hidden)
                node_hidden = _update_GRU(message_vec, node_hidden, reuse_GRU=(i!=0))
                nhs.append(node_hidden)
        
            out = tf.concat(nhs, axis=2)
            
            return out

        def _readout(hidden_0, hidden_n, outdim):    
            
            def _attn_nn(inp, hdim):

                inp = tf.layers.dense(inp, hdim, activation = tf.nn.sigmoid)

                return inp
        
            def _tanh_nn(inp, hdim):

                inp = tf.layers.dense(inp, hdim)

                return inp

            attn_wgt = _attn_nn(tf.concat([hidden_0, hidden_n], 2), aggrdim) 
            tanh_wgt = _tanh_nn(hidden_n, aggrdim)
            readout = tf.reduce_mean(tf.multiply(tanh_wgt, attn_wgt) * mask, 1)
            
            if prop is not None: readout = tf.concat([readout, prop], 1)
            
            readout = tf.nn.tanh(tf.layers.dense(readout, aggrdim))
            readout = tf.nn.tanh(tf.layers.dense(readout, aggrdim))
            pred = tf.layers.dense(readout, outdim) 
    
            return pred

        with tf.variable_scope(name, reuse=reuse):
            
            n_node = int(node.shape[1])

            mask = tf.reduce_max(node[:,:,2+3:], 2, keepdims=True)
            
            edge_wgt = _edge_nn(edge)
            hidden_0 = _embed_node(node)
            hidden_n = _MPNN(edge_wgt, hidden_0, n_step)
            
            readout = _readout(hidden_0, hidden_n, latentdim)	
    	
        return readout
    
    
    def _generator(self, batch_size, latent, n_step, name='', reuse=True):

        def _decoder_node(vec):    
        
            vec = tf.layers.dense(vec, (self.n_node + self.n_dummy) * (self.dim_node + 3) )
            vec = tf.reshape(vec, [batch_size, self.n_node + self.n_dummy, self.dim_node + 3])

            logit1 = vec[:,:,:3]
            probs1 = tf.nn.softmax(logit1)[:,:,:-1]

            logit2 = vec[:,:,3:3+4]
            probs2 = tf.nn.softmax(logit2)[:,:,:-1]

            logit3 = vec[:,:,3+4:]
            probs3 = tf.nn.softmax(logit3)[:,:,:-1]

            probout = tf.concat([probs1, probs2, probs3], 2)
            
            return probout

        def _decoder_edge(vec):    

            vec = tf.layers.dense(vec, (self.n_node + self.n_dummy) * (self.n_node + self.n_dummy) * (self.dim_edge+1))
            vec = tf.reshape(vec, [batch_size, self.n_node + self.n_dummy, self.n_node + self.n_dummy, self.dim_edge+1])
            
            logit = (vec + tf.transpose(vec, perm = [0, 2, 1, 3])) / 2
            
            probs = tf.nn.softmax(logit)[:,:,:,:-1] * tf.reshape(1-tf.eye(self.n_node + self.n_dummy), [1, self.n_node + self.n_dummy, self.n_node + self.n_dummy, 1])
              
            return probs

        with tf.variable_scope(name, reuse=reuse):
        
            for _ in range(n_step):
                latent = tf.nn.leaky_relu(tf.layers.batch_normalization(tf.layers.dense(latent, self.dim_f)))
            
            rec_node_prob = _decoder_node(latent)
            rec_edge_prob = _decoder_edge(latent)

        return rec_node_prob, rec_edge_prob

        
    def _iso_KLD(self, mu, lsgm):
    
        a = tf.exp(lsgm) + tf.square(mu)
        b = 1 + lsgm
    
        kld = 0.5 * tf.reduce_sum(a - b, 1, keepdims = True)
    
        return kld
