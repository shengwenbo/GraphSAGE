from graphsage.layers import Layer,Dense
import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

class NeighborGenerator(Layer):
    def __init__(self, input_dim, output_dim=-1, hidden_dims=[64,32,16], dropout=.0, bias=True, **kwargs):
        super(NeighborGenerator, self).__init__(**kwargs)

        self.input_dim = input_dim
        if output_dim > 0:
            self.output_dim = output_dim
        else:
            self.output_dim = input_dim
        self.hidden_dims = hidden_dims

        self.dropout = dropout
        self.bias = bias

        self._build()

    def _build(self):

        self.hidden_layers = []
        last_dim = self.input_dim
        i = 0
        for dim in self.hidden_dims:
            hidden = Dense(last_dim, dim, act=tf.nn.tanh, dropout=self.dropout, bias=self.bias, name="hidden_%d" % i)
            self.hidden_layers.append(hidden)
            for name, var in hidden.vars.items():
                self.vars["hidden_%d_%s" % (i, name)] = var
            last_dim = dim
            i += 1

        self.output_layer = Dense(last_dim, self.output_dim, act=tf.nn.tanh, dropout=self.dropout, bias=self.bias, name="output")
        for name, var in self.output_layer.vars.items():
            self.vars["output_%s" % name] = var

    def _call(self, inputs):
        """
        Generate neighbors
        :param inputs: [features, num_samples]
                        features: embeddings of center nodes, (batch size, input_dim)
                        num_samples: count of sampled (generated) nodes, int
        :return:
        """
        features, num_samples = inputs
        neighbors = []
        for i in range(num_samples):
            hidden = features
            for layer in self.hidden_layers:
                hidden = layer(hidden)
            neighbors.append(hidden)
        neighbors = tf.concat(neighbors, axis=-1)
        neighbors = tf.reshape(neighbors, [-1, self.hidden_dims[-1]])

        output = self.output_layer(neighbors)
        output = tf.reshape(output, [-1, self.input_dim])

        return output
