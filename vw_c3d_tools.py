
import tensorflow as tf
import numpy as np


#%%
def conv(layer_name, x, out_channels, weight_decay, biases_decay, kernel_size=[3,3,3], stride=[1,1,1,1,1], is_pretrain=True):
    in_channels = x.get_shape()[-1]
    with tf.variable_scope(layer_name):
        w = tf.get_variable(name='weights',
                            trainable=is_pretrain,
                            shape=[kernel_size[0], kernel_size[1], kernel_size[2], in_channels, out_channels],
                            initializer=tf.contrib.layers.xavier_initializer()) # default is uniform distribution initialization
        #collect the loss to the collection "losses"
        if weight_decay is not None:
            weight_loss = tf.multiply(tf.nn.l2_loss(w), weight_decay, name='weight_loss')
            tf.add_to_collection('losses', weight_loss)
            tf.summary.scalar('weight_loss', weight_loss)
        ############################################
        b = tf.get_variable(name='biases',
                            trainable=is_pretrain,
                            shape=[out_channels],
                            initializer=tf.constant_initializer(0.0))
        #collect the loss to the collection "losses"
        if biases_decay is not None:
            biase_loss = tf.multiply(tf.nn.l2_loss(b), biases_decay, name='biase_loss')
            tf.add_to_collection('losses', biase_loss)
            tf.summary.scalar('biase_loss', biase_loss)
        ############################################
        x = tf.nn.conv3d(x, w, stride, padding='SAME', name='conv')
        x = tf.nn.bias_add(x, b, name='bias_add')
        x = tf.nn.relu(x, name='relu')

        return x

#%%
def pool(layer_name, x, k, is_max_pool=True):
    if is_max_pool:
        x = tf.nn.max_pool3d(x, ksize=[1,k,2,2,1], strides=[1,k,2,2,1], padding='SAME',name=layer_name)
        #(x, kernel, strides=stride, padding='SAME', name=layer_name)
    else:
        x = tf.nn.avg_pool3d(x, ksize=[1,k,2,2,1], strides=[1,k,2,2,1], padding='SAME',name=layer_name)
    return x

#%%
def batch_norm(x):
    '''Batch normlization(I didn't include the offset and scale)
    '''
    epsilon = 1e-3
    batch_mean, batch_var = tf.nn.moments(x, [0])
    x = tf.nn.batch_normalization(x,
                                  mean=batch_mean,
                                  variance=batch_var,
                                  offset=None,
                                  scale=None,
                                  variance_epsilon=epsilon)
    return x

#%%
def FC_layer(layer_name, x, out_nodes,weight_decay, biases_decay,_dropout):
    shape = x.get_shape()
    if len(shape) == 5:
        size = shape[1].value * shape[2].value * shape[3].value * shape[4].value
    else:
        size = shape[-1].value

    with tf.variable_scope(layer_name):
        w = tf.get_variable('weights',
                            shape=[size, out_nodes],
                            initializer=tf.contrib.layers.xavier_initializer())
        #collect the loss to the collection "losses"
        if weight_decay is not None:
            weight_loss = tf.multiply(tf.nn.l2_loss(w), weight_decay, name='weight_loss')
            tf.add_to_collection('losses', weight_loss)
            tf.summary.scalar('weight_loss', weight_loss)
        ############################################
        b = tf.get_variable('biases',
                            shape=[out_nodes],
                            initializer=tf.constant_initializer(0.0))
        #collect the loss to the collection "losses"
        if biases_decay is not None:
            biase_loss = tf.multiply(tf.nn.l2_loss(b), biases_decay, name='biase_loss')
            tf.add_to_collection('losses', biase_loss)
            tf.summary.scalar('biase_loss', biase_loss)
        ############################################
        flat_x = tf.reshape(x, [-1, size]) # flatten into 1D
        
        x = tf.nn.bias_add(tf.matmul(flat_x, w), b)
        x = tf.nn.relu(x)
        x = tf.nn.dropout(x, _dropout)
        return x

# #%%
# def loss(logits, labels):
#     '''Compute loss
#     Args:
#         logits: logits tensor, [batch_size, n_classes]
#         labels: one-hot labels
#     '''
#     with tf.name_scope('loss') as scope:
#         cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels,name='cross-entropy')
#         loss = tf.reduce_mean(cross_entropy, name='loss')
#         tf.summary.scalar(scope+'/loss', loss)
#         return loss
#%%
def loss(logits, labels):
    '''Compute loss
    Args:
        logits: logits tensor, [batch_size, n_classes]
        labels: one-hot labels
    '''
    with tf.name_scope('loss') as scope:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels,name='cross-entropy')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.add_to_collection('losses',loss)
        total_loss = tf.add_n(tf.get_collection('losses'),name='total_loss')
        tf.summary.scalar('cross_entropy_loss', loss)
        tf.summary.scalar('total_loss', total_loss)
        return total_loss
    
#%%
def accuracy(logits, labels):
  """Evaluate the quality of the logits at predicting the label.
  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, 
  """
  with tf.name_scope('accuracy') as scope:
      correct = tf.equal(tf.arg_max(logits, 1), tf.arg_max(labels, 1))
      correct = tf.cast(correct, tf.float32)
      accuracy = tf.reduce_mean(correct)*100.0
      tf.summary.scalar(scope+'/accuracy', accuracy)
  return accuracy



#%%
def num_correct_prediction(logits, labels):
  """Evaluate the quality of the logits at predicting the label.
  Return:
      the number of correct predictions
  """
  correct = tf.equal(tf.arg_max(logits, 1), tf.arg_max(labels, 1))
  correct = tf.cast(correct, tf.int32)
  n_correct = tf.reduce_sum(correct)
  return n_correct



#%%
def optimize(loss, learning_rate, global_step):
    '''optimization, use Gradient Descent as default
    '''
    with tf.name_scope('optimizer'):
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op
    


    
# #%%
# def load(data_path, session):
#     data_dict = np.load(data_path, encoding='latin1').item()
    
#     keys = sorted(data_dict.keys())
#     for key in keys:
#         with tf.variable_scope(key, reuse=True):
#             for subkey, data in zip(('weights', 'biases'), data_dict[key]):
#                 session.run(tf.get_variable(subkey).assign(data))
                

# #%%  
# def test_load():
#     data_path = './/vgg16_pretrain//vgg16.npy'
    
#     data_dict = np.load(data_path, encoding='latin1').item()
#     keys = sorted(data_dict.keys())
#     for key in keys:
#         weights = data_dict[key][0]
#         biases = data_dict[key][1]
#         print('\n')
#         print(key)
#         print('weights shape: ', weights.shape)
#         print('biases shape: ', biases.shape)

    
# #%%                
# def load_with_skip(data_path, session, skip_layer):
#     data_dict = np.load(data_path, encoding='latin1').item()
#     for key in data_dict:
#         if key not in skip_layer:
#             with tf.variable_scope(key, reuse=True):
#                 for subkey, data in zip(('weights', 'biases'), data_dict[key]):
#                     session.run(tf.get_variable(subkey).assign(data))

   
# #%%
# def print_all_variables(train_only=True):
#     """Print all trainable and non-trainable variables
#     without tl.layers.initialize_global_variables(sess)

#     Parameters
#     ----------
#     train_only : boolean
#         If True, only print the trainable variables, otherwise, print all variables.
#     """
#     # tvar = tf.trainable_variables() if train_only else tf.all_variables()
#     if train_only:
#         t_vars = tf.trainable_variables()
#         print("  [*] printing trainable variables")
#     else:
#         try: # TF1.0
#             t_vars = tf.global_variables()
#         except: # TF0.12
#             t_vars = tf.all_variables()
#         print("  [*] printing global variables")
#     for idx, v in enumerate(t_vars):
#         print("  var {:3}: {:15}   {}".format(idx, str(v.get_shape()), v.name))   

# #%%   








# ##***** the followings are just for test the tensor size at diferent layers *********##

# #%%
# def weight(kernel_shape, is_uniform = True):
#     ''' weight initializer
#     Args:
#         shape: the shape of weight
#         is_uniform: boolen type.
#                 if True: use uniform distribution initializer
#                 if False: use normal distribution initizalizer
#     Returns:
#         weight tensor
#     '''
#     w = tf.get_variable(name='weights',
#                         shape=kernel_shape,
#                         initializer=tf.contrib.layers.xavier_initializer())    
#     return w

# #%%
# def bias(bias_shape):
#     '''bias initializer
#     '''
#     b = tf.get_variable(name='biases',
#                         shape=bias_shape,
#                         initializer=tf.constant_initializer(0.0))
#     return b

# #%%









    