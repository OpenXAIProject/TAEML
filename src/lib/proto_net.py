import tensorflow as tf
import os
from lib.network import Network

class ProtoNet(Network):
    def __init__(self, name):
        self.name = name

    def classifier(self, sx, qx, is_train, trainable=False,
                   eps=1e-8, scale=1e+1, normalized=True):
        proto_vec = Network.base_net(sx, is_train, reuse=False, trainable=trainable)
        query_vec = Network.base_net(qx, is_train, reuse=True, trainable=trainable)

        if normalized:
            proto_vec = proto_vec /
                    (tf.norm(proto_vec, axis=1, keepdims=True) + 1e-8) * scale
            query_vec = query_vec /
                    (tf.norm(query_vec, axis=1, keepdims=True) + 1e-8) * scale

        dim_ = proto_vec.get_shape().as_list()
        proto_vec = tf.reshape(proto_vec, [nway, -1, dim_[-1]])
        proto_vec = tf.reduce_mean(proto_vec, axis=1)

        _p = tf.expand_dims(proto_vec, axis=0)
        _q = tf.expand_dims(query_vec, axis=1)
        embedding = (_p - _q)**2
        dist = tf.reduce_sum(embedding, axis=2)

        if not normalized:
            dist = dist * 1.

        y = tf.nn.softmax(-dist)

        out_dict = {'pred': y, 'embedding': embedding}
        return out_dict
