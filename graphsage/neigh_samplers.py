from __future__ import division
from __future__ import print_function

from graphsage.layers import Layer,Dense
import tensorflow as tf
import numpy.random as random
flags = tf.app.flags
FLAGS = flags.FLAGS


"""
Classes that are used to sample node neighborhoods
"""

class UniformNeighborSampler(Layer):
    """
    Uniformly samples neighbors.
    Assumes that adj lists are padded with random re-sampling
    """
    def __init__(self, adj_info, **kwargs):
        super(UniformNeighborSampler, self).__init__(**kwargs)
        self.adj_info = adj_info

    def _call(self, inputs):
        ids, num_samples = inputs
        adj_lists = tf.nn.embedding_lookup(self.adj_info, ids) 
        adj_lists = tf.transpose(tf.random_shuffle(tf.transpose(adj_lists)))
        adj_lists = tf.slice(adj_lists, [0,0], [-1, num_samples])
        return adj_lists


class GenerativeNeighborSampler(Layer):
    def __init__(self, input_dim, output_dim=-1, hidden_dims=[64,32,16], dropout=.0, bias=True, **kwargs):
        super(GenerativeNeighborSampler, self).__init__(**kwargs)

        self.input_dim = input_dim
        if output_dim > 0:
            self.output_dim = output_dim
        else:
            self.output_dim = input_dim
        self.hidden_dims = hidden_dims

        self.dropout = dropout
        self.bias = bias

        self.build()

    def _build(self):

        self.hidden_layers = []
        last_dim = self.input_dim
        for dim in self.hidden_dims:
            hidden = Dense(last_dim, dim, act=tf.nn.tanh, dropout=self.dropout, bias=self.bias)
            self.hidden_layers.append(hidden)
            last_dim = dim

        self.output_layer = Dense(last_dim, self.output_dim, act=tf.nn.tanh, dropout=self.dropout, bias=self.bias)

    def _call(self, inputs):
        """
        Generate neighbors
        :param inputs: [features, num_samples]
                        features: embeddings of center nodes, (batch size, input_dim)
                        num_samples: count of sampled (generated) nodes, int
        :return:
        """
        features, num_samples = inputs
        input_mat = tf.stack([tf.tile(fea, num_samples) for fea in features])

        hidden = input_mat
        for layer in self.hidden_layers:
            hidden = layer(hidden)

        output = self.output_layer(hidden)
        output = tf.reshape(output, [-1, num_samples, self.input_dim])

        return output
