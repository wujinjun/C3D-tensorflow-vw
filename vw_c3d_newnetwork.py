import tensorflow as tf
import vw_c3d_tools as c3d_tools

def C3D_MODEL(x,n_classes,is_pretrain=True):
#     conv(layer_name, x, out_channels, weight_decay, biases_decay, kernel_size=[3,3,3], stride=[1,1,1,1,1], is_pretrain=True):
#     pool(layer_name, x, k, is_max_pool=True):
#     FC_layer(layer_name, x, out_nodes,weight_decay, biases_decay):
    conv1 = c3d_tools.conv("conv1", x, 64, 0.0005, 0, is_pretrain=is_pretrain)
    with tf.name_scope('pool1'):    
        pool1 = c3d_tools.pool("pool1", conv1, 1, is_max_pool=True)
    
    conv2 = c3d_tools.conv("conv2", pool1, 128, 0.0005, 0, is_pretrain=is_pretrain)
    with tf.name_scope('pool2'):    
        pool2 = c3d_tools.pool("pool2", conv2, 2, is_max_pool=True)
    
    conv3a = c3d_tools.conv("conv3a", pool2, 256, 0.0005, 0, is_pretrain=is_pretrain)    
    conv3b = c3d_tools.conv("conv3b", conv3a, 256, 0.0005, 0, is_pretrain=is_pretrain)
    with tf.name_scope('pool3'):    
        pool3 = c3d_tools.pool("pool3", conv3b, 2, is_max_pool=True)
    
    conv4a = c3d_tools.conv("conv4a", pool3, 512, 0.0005, 0, is_pretrain=is_pretrain)    
    conv4b = c3d_tools.conv("conv4b", conv4a, 512, 0.0005, 0, is_pretrain=is_pretrain)
    with tf.name_scope('pool4'):    
        pool4 = c3d_tools.pool("pool4", conv4b, 2, is_max_pool=True)

    conv5a = c3d_tools.conv("conv5a", pool4, 512, 0.0005, 0, is_pretrain=is_pretrain)    
    conv5b = c3d_tools.conv("conv5b", conv5a, 512, 0.0005, 0, is_pretrain=is_pretrain)
    with tf.name_scope('pool5'):        
        pool5 = c3d_tools.pool("pool5", conv5b, 2, is_max_pool=True)
    
    pool5 = tf.transpose(pool5, perm=[0,1,4,2,3])
    with tf.name_scope('fc6'):
        fc6 = c3d_tools.FC_layer('fc6', pool5, 4096, 0.0005, 0, 0.5)
        fc6 = c3d_tools.batch_norm(fc6)
    with tf.name_scope('fc7'):
        fc7 = c3d_tools.FC_layer('fc7', fc6, 4096, 0.0005, 0, 0.5)
        fc7 = c3d_tools.batch_norm(fc7)
    with tf.name_scope('fc8'):
        fc8 = c3d_tools.FC_layer('fc8', fc7, n_classes, 0.0005, 0, 1)
    return fc8

def C3D_MODEL_fc6(x,n_classes,is_pretrain=True):
#     conv(layer_name, x, out_channels, weight_decay, biases_decay, kernel_size=[3,3,3], stride=[1,1,1,1,1], is_pretrain=True):
#     pool(layer_name, x, k, is_max_pool=True):
#     FC_layer(layer_name, x, out_nodes,weight_decay, biases_decay):
    conv1 = c3d_tools.conv("conv1", x, 64, 0.0005, 0, is_pretrain=is_pretrain)
    with tf.name_scope('pool1'):    
        pool1 = c3d_tools.pool("pool1", conv1, 1, is_max_pool=True)
    
    conv2 = c3d_tools.conv("conv2", pool1, 128, 0.0005, 0, is_pretrain=is_pretrain)
    with tf.name_scope('pool2'):    
        pool2 = c3d_tools.pool("pool2", conv2, 2, is_max_pool=True)
    
    conv3a = c3d_tools.conv("conv3a", pool2, 256, 0.0005, 0, is_pretrain=is_pretrain)    
    conv3b = c3d_tools.conv("conv3b", conv3a, 256, 0.0005, 0, is_pretrain=is_pretrain)
    with tf.name_scope('pool3'):    
        pool3 = c3d_tools.pool("pool3", conv3b, 2, is_max_pool=True)
    
    conv4a = c3d_tools.conv("conv4a", pool3, 512, 0.0005, 0, is_pretrain=is_pretrain)    
    conv4b = c3d_tools.conv("conv4b", conv4a, 512, 0.0005, 0, is_pretrain=is_pretrain)
    with tf.name_scope('pool4'):    
        pool4 = c3d_tools.pool("pool4", conv4b, 2, is_max_pool=True)

    conv5a = c3d_tools.conv("conv5a", pool4, 512, 0.0005, 0, is_pretrain=is_pretrain)    
    conv5b = c3d_tools.conv("conv5b", conv5a, 512, 0.0005, 0, is_pretrain=is_pretrain)
    with tf.name_scope('pool5'):        
        pool5 = c3d_tools.pool("pool5", conv5b, 2, is_max_pool=True)
    
    pool5 = tf.transpose(pool5, perm=[0,1,4,2,3])
    with tf.name_scope('fc6'):
        fc6 = c3d_tools.FC_layer('fc6', pool5, 4096, 0.0005, 0, 0.5)
        fc6 = c3d_tools.batch_norm(fc6)
    with tf.name_scope('fc7'):
        fc7 = c3d_tools.FC_layer('fc7', fc6, 4096, 0.0005, 0, 0.5)
        fc7 = c3d_tools.batch_norm(fc7)
    with tf.name_scope('fc8'):
        fc8 = c3d_tools.FC_layer('fc8', fc7, n_classes, 0.0005, 0, 1)
    return fc6