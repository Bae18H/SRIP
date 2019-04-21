"""
Monday July 10, 2017

Hyun Jun Bae 

Two-dimensional convolutional neural network
"""

import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as tcl
import os
import pdb

# Limits TensorFlow to using one GPU 
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Clear nodes on graph
tf.reset_default_graph()

# 5x5 filter with stride 2
def conv2d(x, num_input_channels, num_filters,
                   k_w=5, stride_w=2, stddev=0.02, 
                   bias=False, padding='SAME', name='conv2d'): 
        
    W = tf.get_variable(name+'_W', shape=[k_w, k_w, num_input_channels, num_filters], initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv2d(x, W, strides=[1,stride_w,stride_w,1], padding=padding, name=name)
    if bias:
        b = tf.get_variable(name+'_b', [num_filters], initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = conv + b
    return conv
        

class Model(object):
        
    def __init__(self, model_path, train_mode=True, output_dim=27,
                 in_height=64, in_width=64,
                 u_dim=1,
                 num_channels=8,
                 batch_size=100, d_learning_rate=1e-4,
                 measure='mse',
                 ):
        """
        xe means cross-entropy
        mse means mean squared error
        """                                
        self.model_path = model_path
        self.train_mode = train_mode
        self.output_dim = output_dim
        self.in_height = in_height
        self.in_width = in_width
        self.num_channels = num_channels
        self.u_dim = u_dim
        self.measure = measure
                          
        self.batch_size = batch_size

        self._srng = np.random.RandomState(np.random.randint(1,2147462579))
                
        # Initial state
        self.d_loss = 0.0
                
        # Build computation graph of model
        self.x = tf.placeholder(tf.float32, shape=[self.batch_size, self.output_dim])
        self.y = tf.placeholder(tf.float32, shape=[self.batch_size, self.in_height, self.in_width])
        self.ymax = tf.placeholder(tf.float32, shape=[self.batch_size, 1])
        self.pattern = tf.placeholder(tf.float32, shape=[self.batch_size, self.u_dim])

        # y must be a 4D tensor (conv2d); expand_dims can only create a final dimension of 1 (grayscale)
        y = tf.expand_dims(self.y, -1)
        # Creating convolution layers
        h0 = conv2d(y, 1, self.num_channels, k_w=5, stride_w=2, padding="SAME", name="conv0")
        h0 = tf.nn.relu(tcl.batch_norm(h0))
        h1 = conv2d(h0, self.num_channels, 2*self.num_channels, k_w=5, stride_w=2, padding="SAME", name="conv1")
        h1 = tf.nn.relu(tcl.batch_norm(h1))
        h2 = conv2d(h1, 2*self.num_channels, 4*self.num_channels, k_w=5, stride_w=2, padding="SAME", name="conv2")
        h2 = tf.nn.relu(tcl.batch_norm(h2))
        h3 = tcl.flatten(h2)
                
        # Final (Fully connected) layer
        fc = tf.concat([h3, self.ymax, self.pattern], axis=1)
        self.logits = tcl.fully_connected(inputs=fc, num_outputs=self.output_dim, activation_fn=None)
        self.ppred=tf.sigmoid(self.logits)
                
        # Loss function
        if self.measure == 'xe':
            self.d_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.x))            
        elif self.measure == 'mse':
            self.d_loss = tf.reduce_mean(tf.square(self.ppred- self.x))
        else:
            raise NotImplementedError
            
        # List of all trainable variables
        self.d_vars = tf.trainable_variables()
        # Optimizer and autodiff
        self.d_optimizer = tf.train.AdamOptimizer(d_learning_rate, beta1=0.5, beta2=0.999)
        d_grads = self.d_optimizer.compute_gradients(self.d_loss, self.d_vars)
        clip_d_grads = [(tf.clip_by_norm(grad, 5), var) for grad, var in d_grads if grad is not None]
        self.d_optimizer = self.d_optimizer.apply_gradients(clip_d_grads)
        
        
        
    def train(self, train_set, valid_set, maxEpoch=10):        
        
        with tf.Session() as sess:
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            
            #pdb.set_trace()
            
            i = 0
            for epoch in range(maxEpoch): # range for python3
                for xtrain, ytrain, ptrain, pattern in self.data_loader(train_set, self.batch_size, shuffle=True):
                    xtrain = xtrain[:,:,0]
                    #ytrain = ytrain[:,::-1]
                    ptrain = ptrain[:,:,0]
                    pattern = pattern[:,:,0]
                    print(self.x.shape, self.y.shape, self.ymax.shape, self.pattern.shape)
                    print(xtrain.shape, ytrain.shape, ptrain.shape, pattern.shape)
                    pdb.set_trace()
                    _, Ld, param_pred = sess.run([self.d_optimizer, self.d_loss, self.ppred], 
                                                 feed_dict={self.x: xtrain, self.y: ytrain, self.ymax: ptrain, self.pattern: pattern})
                    #pdb.set_trace()
                    i += 1
                    if i % 10 == 0:
                        Ldvs = []
                        for xvalid, yvalid, pvalid, patternv in self.data_loader(valid_set, self.batch_size):
                            xvalid = xvalid[:,:,0]
                            #yvalid = yvalid[:,::-1]
                            pvalid = pvalid[:,:,0]
                            patternv = patternv[:,:,0]
                            Ldv, param_v_pred = sess.run([self.d_loss, self.ppred], feed_dict={self.x: xvalid, self.y: yvalid, self.ymax: pvalid, self.pattern: patternv})
                            Ldvs.append(Ldv)
                        Ld_valid = np.array(Ldvs).mean()
                        print("Iter=%d: Ld: %f  Ld_valid: %f" % (i, Ld, Ld_valid))
                                
                        self.save_model(saver, sess, step=epoch)
                        np.savetxt('nparam_epoch'+str(epoch)+'.txt', xtrain)
                        np.savetxt('nparam_pred_epoch'+str(epoch)+'.txt', param_pred) 
                        np.savetxt('nparam_v'+str(epoch)+'.txt', xvalid)
                        np.savetxt('nparam_v_pred'+str(epoch)+'.txt',param_v_pred)

                
        
    def data_loader(self, dataset, batchsize, shuffle=False): 
        with tf.Session():    
            #pdb.set_trace()
            features, labels, peaks, pattern = dataset
            labels = np.repeat(labels, in_height/3, axis=0)
            labels = np.resize(labels,(in_height,in_height,in_width))
            #print(features.shape, labels.shape, peaks.shape, pattern.shape)
            #pdb.set_trace()
            if shuffle:
                #print(features.get_shape().as_list()[0])
                #pdb.set_trace()
                indices = np.arange(features.get_shape().as_list()[0])
                #pdb.set_trace()
                self._srng.shuffle(indices)
            for start_idx in range(0, features.get_shape().as_list()[0] - batchsize + 1, batchsize):
                if shuffle:
                    excerpt = indices[start_idx:start_idx + batchsize]
                else:
                    excerpt = slice(start_idx, start_idx + batchsize)
                yield features[excerpt], labels[excerpt], peaks[excerpt], pattern[excerpt]
                            
        
    def save_model(self, saver, sess, step):
        """
        save model with path error checking
        """
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        my_path = os.path.join(self.model_path, "my_new_model.ckpt")
        saver.save(sess, os.path.join(os.getcwd(), my_path), global_step=step)



if __name__ == "__main__":
    
    filenames= [('C:/Users/Daniel/Desktop/myc_e2f_data/figureG/gray_myc_ef_2_%d.png' %i) for i in range(1,501)]
    filename_queue = tf.train.string_input_producer(filenames)
    reader = tf.WholeFileReader()
    _, content = reader.read(filename_queue)
    
    image = tf.image.decode_png(content, channels=1)
    image = tf.cast(image, tf.float32)
    result = tf.image.resize_images(image, [64, 64])
    
    #result = tf.train.batch([resized_image], batch_size=100)
    
    train_size=500
    valid_size=300
    
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        #pdb.set_trace()
        in_height=sess.run(result).shape[0]
        in_width=sess.run(result).shape[1]
        #print(in_height, in_width)
        #pdb.set_trace()
        
        train_set = result[:train_size, in_height-2:in_height], result[:train_size, 0:in_width], result[:train_size, -2:-1], result[:train_size, -1:] #parameters,distribution,peak value, pattern
        #print(sess.run(train_set)[0].shape, sess.run(train_set)[1].shape, sess.run(train_set)[2].shape, sess.run(train_set)[3].shape)
        #pdb.set_trace()
        valid_set = result[-valid_size:, in_height-2:in_height], result[-valid_size:, 0:in_width], result[-valid_size:,-2:-1], result[-valid_size:,-1:]
        train_mode = True
        mymodel = Model("save", train_mode=train_mode, output_dim=2, in_height=in_height, in_width=in_width)
        if train_mode == True:
            mymodel.train(train_set, valid_set, maxEpoch=200) # # of iters = maxepoch * N/bs
        
        coord.request_stop()
        coord.join(threads)