import tensorflow as tf

import models as models
import layers as layers
from generator import NeighborGenerator, NeighborGenerator1
from aggregators import MeanAggregator, MaxPoolingAggregator, MeanPoolingAggregator, SeqAggregator, GCNAggregator, AttentionAggregator

flags = tf.app.flags
FLAGS = flags.FLAGS

class SemisupervisedGraphsage(models.SampleAndAggregate):
    """Implementation of supervised GraphSAGE."""

    def __init__(self, num_classes,
            placeholders, features, adj, degrees,
            layer_infos, latent_dim=100, concat=False, aggregator_type="mean",
            model_size="small", sigmoid_loss=False, identity_dim=0,
                **kwargs):
        '''
        Args:
            - placeholders: Stanford TensorFlow placeholder object.
            - features: Numpy array with node features.
            - adj: Numpy array with adjacency lists (padded with random re-samples)
            - degrees: Numpy array with node degrees. 
            - layer_infos: List of SAGEInfo namedtuples that describe the parameters of all 
                   the recursive layers. See SAGEInfo definition above.
            - concat: whether to concatenate during recursive iterations
            - aggregator_type: how to aggregate neighbor information
            - model_size: one of "small" and "big"
            - sigmoid_loss: Set to true if nodes can belong to multiple classes
        '''

        models.GeneralizedModel.__init__(self, **kwargs)

        self.generator_cls = NeighborGenerator1

        if aggregator_type == "mean":
            self.aggregator_cls = MeanAggregator
        elif aggregator_type == "seq":
            self.aggregator_cls = SeqAggregator
        elif aggregator_type == "meanpool":
            self.aggregator_cls = MeanPoolingAggregator
        elif aggregator_type == "maxpool":
            self.aggregator_cls = MaxPoolingAggregator
        elif aggregator_type == "gcn":
            self.aggregator_cls = GCNAggregator
        elif aggregator_type == "attention":
            self.aggregator_cls = AttentionAggregator
        else:
            raise Exception("Unknown aggregator: ", self.aggregator_cls)

        # get info from placeholders...
        self.inputs1 = placeholders["batch_real"]
        self.inputs2 = placeholders["batch_fake"]
        self.mode = placeholders["mode"]
        self.model_size = model_size
        self.latent_dim = latent_dim
        self.adj_info = adj
        if identity_dim > 0:
           self.embeds = tf.get_variable("node_embeddings", [adj.get_shape().as_list()[0], identity_dim])
        else:
           self.embeds = None
        if features is None: 
            if identity_dim == 0:
                raise Exception("Must have a positive value for identity feature dimension if no input features given.")
            self.features = self.embeds
        else:
            self.features = tf.Variable(tf.constant(features, dtype=tf.float32), trainable=False)
            if not self.embeds is None:
                self.features = tf.concat([self.embeds, self.features], axis=1)
        self.degrees = degrees
        self.concat = concat
        self.num_classes = num_classes
        self.sigmoid_loss = sigmoid_loss
        self.dims = [(0 if features is None else features.shape[1]) + identity_dim]
        self.dims.extend([layer_infos[i].output_dim for i in range(len(layer_infos))])
        self.batch_size_real = placeholders["batch_size_real"]
        self.batch_size_fake = placeholders["batch_size_fake"]
        self.placeholders = placeholders
        self.layer_infos = layer_infos

        self.build()


    def build(self):

        # Sampler
        real_samples, support_sizes = self.sample(self.inputs1, self.layer_infos, batch_size=self.batch_size_real)
        self.real_samples = [tf.nn.embedding_lookup(self.features, node_samples) for node_samples in real_samples]
        num_samples = [layer_info.num_samples for layer_info in self.layer_infos]

        # Generator
        self.generated_samples, self.generators = self.generate1(self.layer_infos, batch_size=self.batch_size_fake)

        # Discriminator
        self.outputs_real, self.aggregators, self.hidden_real = self.aggregate_with_feature(self.real_samples, self.dims, num_samples,
                                                                          support_sizes, concat=self.concat, model_size=self.model_size, batch_size=self.batch_size_real)
        self.outputs_fake, self.aggregators, self.hidden_fake = self.aggregate_with_feature(self.generated_samples, self.dims, num_samples,
                                                                          support_sizes, aggregators=self.aggregators, concat=self.concat, model_size=self.model_size, batch_size= self.batch_size_fake)

        self.outputs_real = tf.nn.l2_normalize(self.outputs_real, 1)
        self.outputs_fake = tf.nn.l2_normalize(self.outputs_fake, 1)
        dim_mult = 2 if self.concat else 1
        self.node_pred = layers.Dense(dim_mult*self.dims[-1], self.num_classes+1,
                dropout=self.placeholders['dropout'],
                act=lambda x : x)
        # TF graph management
        self.node_preds_real = self.node_pred(self.outputs_real)
        self.node_preds_fake = self.node_pred(self.outputs_fake)

        # loss
        self._loss()

        # Optimize ops
        self.opt_d_sup = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(self.d_loss_sup + self.d_loss_gen + self.w_loss_d, var_list=self.d_vars)
        self.opt_d_unsup = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(self.d_loss_unsup + self.d_loss_gen + self.w_loss_d, var_list=self.d_vars)
        self.opt_g = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(self.g_loss + self.w_loss_g, var_list=self.g_vars)

        self.preds = self.predict()
        self.final_preds = self.final_predict()

    def _loss(self):

        # Weight decay loss
        self.d_vars = []
        self.g_vars = []
        self.w_loss_d = self.w_loss_g = 0
        for aggregator in self.aggregators:
            for var in aggregator.vars.values():
                self.w_loss_d += FLAGS.weight_decay * tf.nn.l2_loss(var)
                self.d_vars.append(var)
        for var in self.node_pred.vars.values():
            self.w_loss_d += FLAGS.weight_decay * tf.nn.l2_loss(var)
            self.d_vars.append(var)
        for generator in self.generators:
            for var in generator.vars.values():
                # self.w_loss_g += FLAGS.weight_decay * tf.nn.l2_loss(var)
                self.g_vars.append(var)

        # Discriminate loss
        # Supervised: p(y_pred = y_real)
        self.d_loss_sup = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=self.node_preds_real,
                labels=self.placeholders["labels"]
            ))
        # Unsupervised: p(y_pred <> fake)
        self.d_loss_unsup = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=self.node_preds_real,
                labels=self.real_logits(self.batch_size_real)
            ))
        # Generated data: p(y_pred = fake)
        self.d_loss_gen = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=self.node_preds_fake,
                labels=self.fake_logits(self.batch_size_fake)
            ))

        # Generator loss
        self.g_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=self.node_preds_fake,
                labels=self.real_logits(self.batch_size_fake)
            ))
        # self.g_loss += tf.reduce_mean(tf.multiply(self.outputs_real[:self.batch_size_fake] - self.outputs_fake[:self.batch_size_fake],
        #                                           self.outputs_real[:self.batch_size_fake] - self.outputs_fake[:self.batch_size_fake]))
        for hidden_real, hidden_fake in zip(self.real_samples, self.generated_samples):
            self.g_loss += tf.reduce_mean(tf.multiply(hidden_real[:self.batch_size_fake,:] - hidden_fake[:self.batch_size_fake,:],
                                                      hidden_real[:self.batch_size_fake,:] - hidden_fake[:self.batch_size_fake,:]))

        # Total loss
        self.loss = self.w_loss_d + self.w_loss_g + self.d_loss_sup + self.d_loss_unsup + self.d_loss_gen + self.g_loss

        tf.summary.scalar('d_loss_unsup', self.d_loss_unsup + self.w_loss_d)
        tf.summary.scalar('d_loss_sup', self.d_loss_sup + self.w_loss_d)
        tf.summary.scalar('g_loss', self.g_loss + self.w_loss_g)

    def real_logits(self, batch_size):
        return tf.concat([tf.ones(shape=[batch_size, self.num_classes], dtype=tf.float32),
                                 tf.zeros(shape=[batch_size, 1], dtype=tf.float32)], axis=-1)

    def fake_logits(self, batch_size):
        return tf.concat([tf.zeros(shape=[batch_size, self.num_classes], dtype=tf.float32),
                                 tf.ones(shape=[batch_size, 1], dtype=tf.float32)], axis=-1)

    def predict(self):
        return tf.nn.softmax(self.node_preds_real)

    def final_predict(self):
        return tf.nn.softmax(self.node_preds_real[:, :-1])

    def generate1(self, layer_infos, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size_fake
        samples = []
        generators = []
        # size of convolution support at each layer per node
        support_size = 1
        dim = self.features.shape[-1]
        # generate center node
        generator = self.generator_cls(self.latent_dim, 1, output_dim=dim, dropout=self.placeholders["dropout"])
        generators.append(generator)
        samples.append(generator(tf.random_normal([batch_size, self.latent_dim])))
        # generate neighbors
        for k in range(len(layer_infos)):
            t = len(layer_infos) - k - 1
            support_size *= layer_infos[t].num_samples
            generator = self.generator_cls(self.latent_dim, support_size, output_dim=dim, dropout=self.placeholders["dropout"])
            generators.append(generator)
            node = generator(tf.random_normal([batch_size, self.latent_dim]))
            samples.append(node)
        return samples, generators

    def aggregate_with_feature(self, samples, dims, num_samples, support_sizes, batch_size=None,
            aggregators=None, name=None, concat=False, model_size="small"):
        """ At each layer, aggregate hidden representations of neighbors to compute the hidden representations
            at next layer.
        Args:
            samples: a list of samples of variable hops away for convolving at each layer of the
                network. Length is the number of layers + 1. Each is a matrix of node features.
            dims: a list of dimensions of the hidden representations from the input layer to the
                final layer. Length is the number of layers + 1.
            num_samples: list of number of samples for each layer.
            support_sizes: the number of nodes to gather information from for each layer.
            batch_size: the number of inputs (different for batch inputs and negative samples).
        Returns:
            The hidden representation at the final layer for all nodes in batch
        """

        if batch_size is None:
            batch_size = self.batch_size_real

        # length: number of layers + 1
        hidden = samples
        new_agg = aggregators is None
        if new_agg:
            aggregators = []
        for layer in range(len(num_samples)):
            if new_agg:
                dim_mult = 2 if concat and (layer != 0) else 1
                # aggregator at current layer
                if layer == len(num_samples) - 1:
                    aggregator = self.aggregator_cls(dim_mult*dims[layer], dims[layer+1], act=lambda x : x,
                            dropout=self.placeholders['dropout'],
                            name=name, concat=concat, model_size=model_size)
                else:
                    aggregator = self.aggregator_cls(dim_mult*dims[layer], dims[layer+1], act=tf.nn.leaky_relu,
                            dropout=self.placeholders['dropout'],
                            name=name, concat=concat, model_size=model_size)
                aggregators.append(aggregator)
            else:
                aggregator = aggregators[layer]
            # hidden representation at current layer for all support nodes that are various hops away
            next_hidden = []
            # as layer increases, the number of support nodes needed decreases
            for hop in range(len(num_samples) - layer):
                dim_mult = 2 if concat and (layer != 0) else 1
                neigh_dims = [batch_size * support_sizes[hop],
                              num_samples[len(num_samples) - hop - 1],
                              dim_mult*dims[layer]]
                h = aggregator((hidden[hop],
                                tf.reshape(hidden[hop + 1], neigh_dims)))
                next_hidden.append(h)
            hidden = next_hidden
        return hidden[0], aggregators, hidden

