import tensorflow as tf
import math

def model(name):
    with tf.variable_scope(name):
        # keep prob
        #self.keep_prob = tf.placeholder(tf.float32)
            
        """
        placeholders
        """
        X = tf.placeholder(tf.float32, [None, 187, 1], name='input')  
        Y = tf.placeholder(tf.float32, [None, 5], name='output')
        
        X_signal = tf.reshape(X, [-1, 187, 1], name='signal')
        
        global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
        learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        
        
        
        """
        build model
        """
        
        # conv, input_shape: [None, 187, 1]
        C = tf.layers.conv1d(       
                        inputs=X_signal,
                        filters=32,
                        kernel_size=5,
                        strides=1,
                        activation=tf.nn.relu
            )
        #print("C shape:", C.shape)
        # output_shape: [None, 183, 32]
            
        # 【1】 block
        # conv + relu, input_shape: [None, 183, 32]
        C11 = tf.layers.conv1d(
                        inputs=C,
                        filters=32,
                        kernel_size=5,
                        strides=1,
                        padding='SAME',
                        activation=tf.nn.relu
            )
        # output_shape: [None, 183, 32]
        
        # conv, input_shape: [None, 183, 32]
        C12 = tf.layers.conv1d(
                        inputs=C11,
                        filters=32,
                        kernel_size=5,
                        strides=1,
                        padding='SAME',
                        activation=tf.nn.relu
            )
        # output_shape: [None, 183, 32]
        
        #print(C12.shape)
        # shortcut 
        S11 = C12 + C
        # pooling
        M11 = tf.layers.max_pooling1d(S11, pool_size=5, strides=2) # output shape: [None, 90, 32]
        print ('M11.shape: ', M11.get_shape().as_list())    
        # 【2】 block
        # conv + relu
        C21 = tf.layers.conv1d(
                        inputs=M11,
                        filters=32,
                        kernel_size=5,
                        strides=1,
                        padding='SAME',
                        activation=tf.nn.relu
            )
        # conv
        C22 = tf.layers.conv1d(
                        inputs=C21,
                        filters=32,
                        kernel_size=5,
                        strides=1,
                        padding='SAME',
                        activation=tf.nn.relu
            )
        # shortcut 
        S21 = C22 + M11
        # pooling
        M21 = tf.layers.max_pooling1d(S21, pool_size=5, strides=2)   # output shape: [None, 43, 32]
        print ('M21.shape: ', M21.get_shape().as_list())    
        # 【3】 block
        # conv + relu
        C31 = tf.layers.conv1d(
                        inputs=M21,
                        filters=32,
                        kernel_size=5,
                        strides=1,
                        padding='SAME',
                        activation=tf.nn.relu
            )
        # conv
        C32 = tf.layers.conv1d(
                        inputs=C31,
                        filters=32,
                        kernel_size=5,
                        strides=1,
                        padding='SAME',
                        activation=tf.nn.relu
            )
        # shortcut 
        S31 = C32 + M21
        # pooling
        M31 = tf.layers.max_pooling1d(S31, pool_size=5, strides=2)  
        print ('M31.shape: ', M31.get_shape().as_list())
            
        # 【4】 block
        # conv + relu
        C41 = tf.layers.conv1d(
                        inputs=M31,
                        filters=32,
                        kernel_size=5,
                        strides=1,
                        padding='SAME',
                        activation=tf.nn.relu
            )
            # conv
        C42 = tf.layers.conv1d(
                        inputs=C41,
                        filters=32,
                        kernel_size=5,
                        strides=1,
                        padding='SAME',
                        activation=tf.nn.relu
            )
        # shortcut
        S41 = C42 + M31
        # pooling
        M41 = tf.layers.max_pooling1d(S41, pool_size=5, strides=2)    
        print ('M41.shape: ', M41.get_shape().as_list())    
        # 【5】 block
        # conv + relu
        C51 = tf.layers.conv1d(
                        inputs=M41,
                        filters=32,
                        kernel_size=5,
                        strides=1,
                        padding='SAME',
                        activation=tf.nn.relu
            )
        # conv
        C52 = tf.layers.conv1d(
                        inputs=C51,
                        filters=32,
                        kernel_size=5,
                        strides=1,
                        padding='SAME',
                        activation=tf.nn.relu
            )
        # shortcut
        S51 = C52 + M41
        # pooling
        M51 = tf.layers.max_pooling1d(S51, pool_size=5, strides=1, padding='SAME')    
        print ('M51.shape: ', M51.get_shape().as_list())
        # flatten
        F1 = tf.reshape(M51, [-1, M51.get_shape().as_list()[1]*M51.get_shape().as_list()[2]])
        print ('F1.shape: ', F1.get_shape().as_list())    
        # FC + relu
        D1 = tf.layers.dense(inputs=F1, units=32, activation=tf.nn.relu)
        print ('D1.shape: ', D1.get_shape().as_list()) 
        # FC
        D2 = tf.layers.dense(inputs=D1, units=32)
        print ('D2.shape: ', D2.get_shape().as_list()) 
        # output
        output = tf.layers.dense(inputs=D2, units=5, activation=tf.nn.softmax)
        print ('OUTPUT.shape: ', output.get_shape().as_list()) 
        y_pred_cls = tf.argmax(output, axis=1)
        
        return X, Y, output, y_pred_cls, global_step, learning_rate
    
def lr_decay(epoch, n_obs, batch_size):
    initial_lrate = 0.001
    k = 0.75
    t = n_obs// (1000*batch_size)
    lrate = initial_lrate * math.exp(-k*t)
    return lrate