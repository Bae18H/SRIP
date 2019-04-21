# -*- coding: utf-8 -*-
"""
Created on Mon May, 31 2017

@author: Shangying

this code is used for prediction of PDE parameters from 1d distribution.
"""

import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as tcl
import csv
import os

#import pdb

os.environ["CUDA_VISIBLE_DEVICES"]="0"


def conv1d(x, x_filters, n_filers,
                   k_w=5, stride_w=2, stddev=0.02, 
                   bias=False, padding='SAME', name='conv1d'):
        #pdb.set_trace()
        #x = tf.expand_dims(x, -1)
        W = tf.get_variable(name+'W', [k_w, x_filters, n_filers], initializer=tf.truncated_normal_initializer(stddev=stddev))
        #pdb.set_trace()
        #x = tf.expand_dims(x, -1)
        conv = tf.nn.conv1d(x, W, stride=stride_w, padding=padding, name=name)
        if bias:
#        b = tf.get_variable('b', [n_filers], initializer=tf.constant_initializer(0.0))
                b = tf.get_variable(name+'b', [n_filers], initializer=tf.truncated_normal_initializer(stddev=stddev))
                conv = conv + b
        return conv
        

class Model(object):
        
        def __init__(self, model_path, train_mode=True, output_dim=19, T=150,
                                 u_dim=1,
                                 n_df=8,
                                 batch_size=100, d_learning_rate=1e-4, # 3e-5
                                 measure='mse', # mse
                                 ):
                """
                xe means cross-entropy
                mse means mean squared error
                """
                                        
                self.model_path = model_path
                self.train_mode = train_mode
                self.output_dim = output_dim
                self.T = T
                self.n_df = n_df
                self.u_dim = u_dim
                self.measure = measure
                          
                self.batch_size = batch_size

                self._srng = np.random.RandomState(np.random.randint(1,2147462579))
                
                # initial state
                self.d_loss = 0.0
                
                # build computation graph of model
                self.x = tf.placeholder(tf.float32, shape=[self.batch_size, self.output_dim])
                self.y = tf.placeholder(tf.float32, shape=[self.batch_size, self.T])
                self.ymax = tf.placeholder(tf.float32, shape=[self.batch_size, 1])
                self.pattern = tf.placeholder(tf.float32, shape=[self.batch_size, self.u_dim])
               
                y = tf.expand_dims(self.y, -1) 
                h0 = conv1d(y, 1, self.n_df, k_w=5, stride_w=2, padding="SAME", name="conv0") # bs x T/2 x ndf
                h0 = tf.nn.relu(tcl.batch_norm(h0))
                h1 = conv1d(h0, self.n_df, 2*self.n_df, k_w=5, stride_w=2, padding="SAME", name="conv1") # bs x T/4 x 2ndf
                h1 = tf.nn.relu(tcl.batch_norm(h1))
                h2 = conv1d(h1, 2*self.n_df, 4*self.n_df, k_w=5, stride_w=2, padding="SAME", name="conv2") # bs x T/8 x 4ndf
                h2 = tf.nn.relu(tcl.batch_norm(h2))
                h3 = tcl.flatten(h2) # bs x (T*ndf/2)
                

                fc = tf.concat([h3, self.ymax, self.pattern], axis=1)
                self.logits = tcl.fully_connected(inputs=fc, num_outputs=self.output_dim, activation_fn=None)
                self.ppred=tf.sigmoid(self.logits)
                #pdb.set_trace()
                
                if self.measure == 'xe':
                        self.d_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.x))            
                elif self.measure == 'mse':
                        self.d_loss = tf.reduce_mean(tf.square(self.ppred- self.x))
                else:
                        raise NotImplementedError
                
                self.d_vars = tf.trainable_variables()

                self.d_optimizer = tf.train.AdamOptimizer(d_learning_rate, beta1=0.5, beta2=0.999)
                d_grads = self.d_optimizer.compute_gradients(self.d_loss, self.d_vars)
                clip_d_grads = [(tf.clip_by_norm(grad, 5), var) for grad, var in d_grads if grad is not None]
                self.d_optimizer = self.d_optimizer.apply_gradients(clip_d_grads)
                
        
        def train(self, train_set, valid_set, maxEpoch=10):
                
                with tf.Session() as sess:
                        
                        saver = tf.train.Saver()
                        sess.run(tf.global_variables_initializer())
                        
                        i = 0
                        for epoch in range(maxEpoch): # range for python3
                                
                                for xtrain, ytrain, ptrain, pattern in self.data_loader(train_set, self.batch_size, shuffle=True):
                                        ytrain = ytrain[:,::-1]
                                       
                                        #pdb.set_trace()
                                        _, Ld, param_pred = sess.run([self.d_optimizer, self.d_loss, self.ppred], feed_dict={self.x: xtrain, self.y: ytrain, 
                                                                                                self.ymax: ptrain, 
                                                                                                self.pattern: pattern})
                                        i += 1
                                        if i % 10 == 0:
                                            Ldvs = []
                                            for xvalid, yvalid, pvalid, patternv in self.data_loader(valid_set, self.batch_size):
                                                yvalid = yvalid[:,::-1]
                                                Ldv, param_v_pred = sess.run([self.d_loss, self.ppred], feed_dict={self.x: xvalid, self.y: yvalid, self.ymax: pvalid, self.pattern: patternv})
                                                Ldvs.append(Ldv)
                                            Ld_valid = np.array(Ldvs).mean()
                                            print("Iter=%d: Ld: %f  Ld_valid: %f" % (i, Ld, Ld_valid))
                                
                                self.save_model(saver, sess, step=epoch)
                                np.savetxt('param_epoch'+str(epoch)+'.txt', xtrain )
                                np.savetxt('param_pred_epoch'+str(epoch)+'.txt', param_pred ) 
                                np.savetxt('param_v'+str(epoch)+'.txt',xvalid )
                                np.savetxt('param_v_pred'+str(epoch)+'.txt',param_v_pred )
                        

                
        
        def data_loader(self, dataset, batchsize, shuffle=False): 
                features, labels, peaks, pattern = dataset
                if shuffle:
                        indices = np.arange(len(features))
                        self._srng.shuffle(indices)
                for start_idx in range(0, len(features) - batchsize + 1, batchsize):
                        if shuffle:
                                excerpt = indices[start_idx:start_idx + batchsize]
                        else:
                                excerpt = slice(start_idx, start_idx + batchsize)
                        yield features[excerpt], labels[excerpt], peaks[excerpt], pattern[excerpt]
                        
        
        def save_model(self, saver, sess, step):
                """
                save model with path error checking
                """
                if self.model_path is None:
                        my_path = "save" # default path in tensorflow saveV2 format
                        # try to make directory if "save" path does not exist
                        if not os.path.exists("save"):
                                try:
                                        os.makedirs("save")
                                except OSError as e:
                                        if e.errno != errno.EEXIST:
                                                raise
                else: 
                        my_path = self.model_path + "/mymodel"
                                
                saver.save(sess, my_path, global_step=step)
                

if __name__ == "__main__":
        
        
    # Load data from csv file (the data file needs to be shuffled before loading for best performance)
    with open('ColonyESBL1D_aCaB_modified2.csv') as csvfile:
        mpg = list(csv.reader(csvfile))
        results = np.array(mpg).astype("float")

    train_size=5000
    valid_size=3000
    
    train_set = results[:train_size,150:152], results[:train_size,0:150], results[:train_size,-2:-1], results[:train_size,-1:] #parameters,distribution,peak value, pattern
    valid_set = results[-valid_size:,150:152], results[-valid_size:,0:150], results[-valid_size:,-2:-1], results[-valid_size:,-1:]
    train_mode = True
    mymodel = Model("save", train_mode=train_mode, output_dim=2, T=150)
    if train_mode == True:
        mymodel.train(train_set, valid_set, maxEpoch=200) # # of iters = maxepoch * N/bs
