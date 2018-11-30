import tensorflow as tf
import os

def taeml_net(name, in_x, nway, scaler=1e+2):
    with tf.variable_scope(name):
        tl = tf.layers.conv2d(in_x, 512, kernel_size=[1,nway], padding='VALID')
        tl = tf.nn.relu(tl)
        tl = tf.layers.conv2d(tl, 128, kernel_size=[1,1], padding='SAME')
        tl = tf.nn.relu(tl)
        tl = tf.reduce_mean(tl, axis=(1,2))
        tl = tf.layers.dense(tl, 1)
        fe = tf.nn.sigmoid(tl) * scaler
        #pemb = tf.nn.softmax(tf.squeeze(pemb))
    return fe

def taeml_net_nontrans(name, in_x, nway, scaler=1e+2):
    with tf.variable_scope(name):
        tl = tf.layers.conv2d(in_x, 512, kernel_size=[1,nway], padding='VALID',
                              data_format='channels_first')
        tl = tf.nn.relu(tl)
        tl = tf.layers.conv2d(tl, 128, kernel_size=[1,1], padding='SAME',
                              data_format='channels_first')
        tl = tf.nn.relu(tl)
        shape = tl.get_shape().as_list()
        tl = tf.reshape(tl, [-1,shape[-1]])
        tl = tf.layers.dense(tl, 1)
        fe = tf.nn.sigmoid(tl) * scaler
    return fe

def cross_entropy(pred, label):
    return -tf.reduce_mean(tf.reduce_sum(label*tf.log(pred+1e-10), axis=1))

def tf_acc(p, y):
    acc = tf.equal(tf.argmax(y,1), tf.argmax(p,1))
    acc = tf.reduce_mean(tf.cast(acc, 'float'))
    return acc
