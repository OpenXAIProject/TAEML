import numpy as np
import sys
import tensorflow as tf
from lib.networks import Network
from utils.utils import cross_entropy, tf_acc

class MAML(Network):
    def __init__(self, config):
        self.update_lr          =  config['update_lr']
        self.test_num_updates   =  config['test_num_updates']
        self.num_updates        =  config['num_updates']
        self.is_train           =  config['is_train']
        self.num_classes        =  config['num_classes']

        self.weights            =  self._build_graph()

    def _build_graph(self):
        weights = {}

        dtype = tf.float32
        conv_initializer =  tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
        fc_initializer =  tf.contrib.layers.xavier_initializer(dtype=dtype)
        k = 3

        with tf.variable_scope('model', reuse=False):
            weights['conv1'] = tf.get_variable('conv1', [k, k, 3, 64],
                                    initializer=conv_initializer, dtype=dtype)
            weights['b1'] = tf.Variable(tf.zeros([64]))
            weights['conv2'] = tf.get_variable('conv2', [k, k, 64, 64],
                                    initializer=conv_initializer, dtype=dtype)
            weights['b2'] = tf.Variable(tf.zeros([64]))
            weights['conv3'] = tf.get_variable('conv3', [k, k, 64, 64],
                                    initializer=conv_initializer, dtype=dtype)
            weights['b3'] = tf.Variable(tf.zeros([64]))
            weights['conv4'] = tf.get_variable('conv4', [k, k, 64, 64],
                                    initializer=conv_initializer, dtype=dtype)
            weights['b4'] = tf.Variable(tf.zeros([64]))
            weights['w5'] = tf.get_variable('w5', [64*5*5, self.num_classes],
                                    initializer=fc_initializer)
            weights['b5'] = tf.Variable(tf.zeros([self.num_classes]), name='b5')
        return weights

    def _conv_block(self, x, kernel, bias, name, reuse):
        with tf.variable_scope(name, reuse=reuse):
            x = tf.nn.conv2d(x, kernel, strides=[1,1,1,1], padding='SAME', data_format='NHWC')
            x = tf.nn.bias_add(x, bias, data_format='NHWC')
            x = tf.contrib.layers.batch_norm(x, data_format='NHWC', fused=False,
                                             is_training=self.is_train, scale=True, center=True)
            x = tf.nn.relu(x)
            x = tf.contrib.layers.max_pool2d(x, 2, data_format='NHWC')

        return x

    def _classifier(self, x, weights, reuse):
        x = self._conv_block(x, weights['conv1'], weights['b1'], name='block_1', reuse=reuse)
        x = self._conv_block(x, weights['conv2'], weights['b2'], name='block_2', reuse=reuse)
        x = self._conv_block(x, weights['conv3'], weights['b3'], name='block_3', reuse=reuse)
        x = self._conv_block(x, weights['conv4'], weights['b4'], name='block_4', reuse=reuse)
        x = tf.layers.flatten(x)
        x = tf.matmul(x, weights['w5']) + weights['b5']

        return x

    def train(self, inputa, inputb, labela, labelb):

        #def _task_metalearn(inputa, inputb, labela, labelb):
        def _task_metalearn(inp):
            inputa, inputb, labela, labelb = inp
            weights = self.weights
            """ Perform gradient descent for one task in the meta-batch. """
            task_outputbs, task_lossesb, task_accuraciesb = [], [], []

            task_outputa = self._classifier(inputa, weights, reuse=False)
            print(inputa.shape, labela.shape)
            task_lossa = cross_entropy(tf.nn.softmax(task_outputa), labela)

            grads = tf.gradients(task_lossa, list(weights.values()))
            #if FLAGS.stop_grad:
            #    grads = [tf.stop_gradient(grad) for grad in grads]
            gradients = dict(zip(weights.keys(), grads))
            fast_weights = dict(zip(weights.keys(),
                                [weights[key] - self.update_lr*gradients[key]
                                for key in weights.keys()]))
            output = self._classifier(inputb, fast_weights, reuse=True)
            task_outputbs.append(output)
            task_lossesb.append(cross_entropy(tf.nn.softmax(output), labelb))

            for j in range(num_updates - 1):
                loss = cross_entropy(tf.nn.softmax(self._classifier(inputa, fast_weights, reuse=True)), labela)
                grads = tf.gradients(loss, list(fast_weights.values()))
                #if FLAGS.stop_grad:
                #    grads = [tf.stop_gradient(grad) for grad in grads]
                gradients = dict(zip(fast_weights.keys(), grads))
                fast_weights = dict(zip(fast_weights.keys(),
                                    [fast_weights[key] -
                                     self.update_lr*gradients[key]
                                    for key in fast_weights.keys()]))
                output = self._classifier(inputb, fast_weights, reuse=True)
                task_outputbs.append(output)
                task_lossesb.append(cross_entropy(tf.nn.softmax(output), labelb))

            task_output = [task_outputa, task_outputbs, task_lossa, task_lossesb]

            task_accuracya = tf.contrib.metrics.accuracy(
                                tf.argmax(tf.nn.softmax(task_outputa), 1),
                                tf.argmax(labela, 1))
            for j in range(num_updates):
                task_accuraciesb.append(tf.contrib.metrics.accuracy(
                                tf.argmax(tf.nn.softmax(task_outputbs[j]), 1),
                                tf.argmax(labelb, 1)))
            task_output.extend([task_accuracya, task_accuraciesb])

            return task_output

        num_updates = self.num_updates
        meta_batch_size = int(inputa.shape[0])

        lossesa, outputas, accuraciesa = [], [], []
        lossesb, outputbs, accuraciesb = [[]]*num_updates, [[]]*num_updates, [[]]*num_updates

        out_dtype = [tf.float32, [tf.float32]*num_updates,
                     tf.float32, [tf.float32]*num_updates]
        out_dtype.extend([tf.float32, [tf.float32]*num_updates])

        result = tf.map_fn(_task_metalearn,
                    elems=(inputa, inputb, labela, labelb),
                    dtype=out_dtype, parallel_iterations=meta_batch_size)
        outputas, outputbs, lossesa, lossesb, accuraciesa, accuraciesb = result

        total_loss1 = tf.reduce_mean(lossesa)
        total_losses2 = [tf.reduce_mean(lossesb[j]) for j in range(num_updates)]

        total_accuracy1 = tf.reduce_mean(accuraciesa)
        total_accuracies2 = [tf.reduce_mean(accuraciesb[j])
                             for j in range(num_updates)]

        print(outputbs)
        print(total_losses2, total_accuracies2)

        return total_losses2[-1], total_accuracies2[-1], outputbs[-1]
