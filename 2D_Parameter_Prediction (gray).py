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
from datetime import datetime
from scipy import io

# Creates timestamp
now = datetime.utcnow().strftime('%Y%m%d_%H%M%S')

# Setup log directory for TensorBoard
root_logdir = "tf_logs"
logdir = "{}/run-{}".format(root_logdir, now)

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
        with tf.name_scope('nodes'):
            self.x = tf.zeros([self.batch_size, self.output_dim], tf.float32)
            self.y = tf.zeros([self.batch_size, self.in_height, self.in_width,1], tf.float32)
            self.ymax = tf.zeros([self.batch_size, 1, 1], tf.float32)
            self.pattern = tf.zeros([self.batch_size, self.u_dim ,self.u_dim], tf.float32)
        
        y = self.y
        # Creating convolution layers
        with tf.name_scope('hidden'):
            h0 = conv2d(y, 1, self.num_channels, k_w=5, stride_w=2, padding="SAME", name="conv0")
            h0 = tf.nn.relu(tcl.batch_norm(h0))
            h1 = conv2d(h0, self.num_channels, 2*self.num_channels, k_w=5, stride_w=2, padding="SAME", name="conv1")
            h1 = tf.nn.relu(tcl.batch_norm(h1))
            h2 = conv2d(h1, 2*self.num_channels, 4*self.num_channels, k_w=5, stride_w=2, padding="SAME", name="conv2")
            h2 = tf.nn.relu(tcl.batch_norm(h2))
            h3 = tcl.flatten(h2)
                
        # Final (Fully connected) layer
        with tf.name_scope('fully_connected'):
            fc = tf.concat([h3, tcl.flatten(self.ymax), tcl.flatten(self.pattern)], axis=1)
            self.logits = tcl.fully_connected(inputs=fc, num_outputs=self.output_dim, activation_fn=None)
            self.ppred=tf.sigmoid(self.logits)
                
        # Loss function
        with tf.name_scope('loss'):
            if self.measure == 'xe':
                self.d_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.x))            
            elif self.measure == 'mse':
                self.d_loss = tf.reduce_mean(tf.square(self.ppred- self.x))
            else:
                raise NotImplementedError
        
        with tf.name_scope('optimize'):
            # List of all trainable variables
            self.d_vars = tf.trainable_variables()
            # Optimizer and autodiff
            self.d_optimizer = tf.train.AdamOptimizer(d_learning_rate, beta1=0.5, beta2=0.999)
            d_grads = self.d_optimizer.compute_gradients(self.d_loss, self.d_vars)
            clip_d_grads = [(tf.clip_by_norm(grad, 5), var) for grad, var in d_grads if grad is not None]
            self.d_optimizer = self.d_optimizer.apply_gradients(clip_d_grads)
        
        # Create summaries for TensorBoard
        if self.measure == 'xe':
            self.loss_summary = tf.summary.scalar('XE', self.d_loss)
        else:
            self.loss_summary = tf.summary.scalar('MSE', self.d_loss)
        self.file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
        
        
        
    def train(self, train_set, valid_set, maxEpoch=10):        
        
        with tf.Session() as sess:
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            #pdb.set_trace()
            
            i = 0
            for epoch in range(maxEpoch): # range for python3
                xtrain, ytrain, ptrain, pattern = train_set
                #print(self.x.shape, self.y.shape, self.ymax.shape, self.pattern.shape)
                #print(xtrain.shape, ytrain.shape, ptrain.shape, pattern.shape)
                #pdb.set_trace()
                _, Ld, param_pred = sess.run([self.d_optimizer, self.d_loss, self.ppred], 
                                             feed_dict={self.x: xtrain.eval(), self.y: ytrain.eval(), self.ymax: ptrain.eval(), self.pattern: pattern.eval()})
                #print("Iteration %d" %i)
                #pdb.set_trace()
                i += 1
                if i % 10 == 0:
                    Ldvs = []
                    xvalid, yvalid, pvalid, patternv = valid_set
                    Ldv, param_v_pred = sess.run([self.d_loss, self.ppred],
                                                 feed_dict={self.x: xvalid.eval(), self.y: yvalid.eval(), self.ymax: pvalid.eval(), self.pattern: patternv.eval()})
                    Ldvs.append(Ldv)
                    Ld_valid = np.array(Ldvs).mean()
                    print("Iter=%d: Ld: %f  Ld_valid: %f" % (i, Ld, Ld_valid))
                                
                    self.save_model(saver, sess, step=epoch)
                    #pdb.set_trace()
                    txt_path = os.path.join(self.model_path, 'Parameters')
                    txt_path = os.path.join(os.getcwd(), txt_path)
                    if not os.path.exists(txt_path):
                        os.makedirs(txt_path)
                    np.savetxt(os.path.join(txt_path, 'nparam_epoch'+str(epoch)+'.txt'), xtrain.eval())
                    np.savetxt(os.path.join(txt_path, 'nparam_pred_epoch'+str(epoch)+'.txt'), param_pred) 
                    np.savetxt(os.path.join(txt_path, 'nparam_v'+str(epoch)+'.txt'), xvalid.eval())
                    np.savetxt(os.path.join(txt_path, 'nparam_v_pred'+str(epoch)+'.txt'),param_v_pred)
                    
                    summary_str = self.loss_summary.eval(feed_dict={self.x: xvalid.eval(), self.y: yvalid.eval()})
                    self.file_writer.add_summary(summary_str, epoch)
                    
            coord.request_stop()
            coord.join(threads)
            self.file_writer.close()
                
   
    def save_model(self, saver, sess, step):
        """
        save model with path error checking
        """
        if not os.path.exists(os.path.join(self.model_path, 'Checkpoints')):
            os.makedirs(os.path.join(self.model_path, 'Checkpoints'))
        my_path = os.path.join(self.model_path, 'Checkpoints\my_model.ckpt')
        saver.save(sess, os.path.join(os.getcwd(), my_path), global_step=step)



if __name__ == "__main__":
    
    with tf.name_scope('input'):
        filenames= [('C:/Users/Daniel/Desktop/myc_e2f_data/figureG/gray_myc_ef_2_%d.png' %i) for i in range(1,501)]
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=None)
        reader = tf.WholeFileReader()
        _, content = reader.read(filename_queue)
        
        image = tf.image.decode_png(content, channels=1)
        image = tf.cast(image, tf.float32)
        resized_image= tf.image.resize_images(image, [64, 64])
        
        result = tf.train.shuffle_batch([resized_image], batch_size=300, capacity=500, min_after_dequeue=100)
        sresult, _ = tf.nn.top_k(result)
        
        mat_data = io.loadmat('C:/Users/Daniel/Desktop/myc_e2f_data/data_myc_ef_yao_2')
        mat_result = mat_data['paraset_list']
        presult = tf.convert_to_tensor(mat_result)[0:300]
        presult = tf.random_shuffle(presult)
        
    #print(result)
    #pdb.set_trace()
    
    train_size=500
    valid_size=300
    
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        in_height=sess.run(result).shape[1]
        in_width=sess.run(result).shape[2]
        par_dim = presult.shape[1]
        train_mode = True
        #print(in_height, in_width)
        #pdb.set_trace()
        
        train_set = presult[:train_size, par_dim-2:par_dim], result[:train_size, 0:in_width, 0:in_height], sresult[:train_size, -2:-1, -2:-1, 0], sresult[:train_size, -1:, -1:, 0] #parameters, distribution, peak value, pattern
        #print(sess.run(train_set)[0].shape, sess.run(train_set)[1].shape, sess.run(train_set)[2].shape, sess.run(train_set)[3].shape)
        #pdb.set_trace()
        valid_set = presult[-valid_size:, par_dim-2:par_dim], result[-valid_size:, 0:in_width, 0:in_height], sresult[-valid_size:, -2:-1, -2:-1, 0], sresult[-valid_size:,-1:, -1:, 0]        
        
        mymodel = Model(now+' Save Data', train_mode=train_mode, output_dim=2, in_height=in_height, in_width=in_width,batch_size=300)
        if train_mode == True:
            try:
                if coord.should_stop():
                    coord.request_stop()
                    coord.join(threads)
                mymodel.train(train_set, valid_set, maxEpoch=200) # # of iters = maxepoch * N/bs
            except Exception as e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join(threads)
        
        # Run TensorBoard
        log_path=os.path.join(os.getcwd(), logdir)
        os.system("tensorboard --logdir "+log_path+" --host=127.0.0.1")