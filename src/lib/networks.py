import tensorflow as tf
import os

class Network():
    def __init__(self, name='all', root=None):
        if root is None:
            self.model_loc = 'models/' + name + '/model.ckpt'
        else:
            self.model_loc = os.path.join(root, name, 'model.ckpt')
        self.name = name
        self.vars = {}

    def conv_block(self, x, name, reuse, isTr, trainable=True, channel=64):
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            l = tf.layers.conv2d(x, channel, kernel_size=3, padding='SAME',
                    trainable=trainable, data_format='channels_first',
                    kernel_initializer=tf.truncated_normal_initializer(.0, stddev=.001))
            l = tf.contrib.layers.batch_norm(l, is_training=isTr, decay=.99,
                    scale=True, center=True, trainable=trainable, fused=True,
                    data_format='NCHW')
            l = tf.nn.relu(l)
            l = tf.contrib.layers.max_pool2d(l, 2, data_format='NCHW')
        return l

    def global_avg_pool(self, x):
        return tf.reduce_mean(x, axis=(1,2)) # only works for NHWC

    def fc(self, in_x, out_dim):
        return tf.layers.dense(in_x, out_dim)

    def base_net(self, in_x, isTr, reuse=False, trainable=True):
        with tf.variable_scope(self.name):
            l = self.conv_block(in_x, 'c1', reuse, isTr, trainable)
            l = self.conv_block(l, 'c2', reuse, isTr, trainable)
            l = self.conv_block(l, 'c3', reuse, isTr, trainable)
            l = self.conv_block(l, 'c4', reuse, isTr, trainable)
            l = tf.layers.flatten(l)
        return l

    def save(self, saver, sess, loc=None):
        if loc is None:
            print ('saved at : ', self.model_loc)
            saver.save(sess, self.model_loc)
        else:
            print ('saved at : ', loc)
            saver.save(sess, loc)

    def restore(self, saver, sess, loc=None):
        if loc is None:
            print ('loaded from : ', self.model_loc)
            saver.restore(sess, self.model_loc)
        else:
            print ('loaded from : ', loc)
            saver.restore(sess, loc)
