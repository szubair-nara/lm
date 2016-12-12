import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
from tensorflow.python.ops import nn_ops

class ConcatNext():

    def __init__(self, batch_size=128,
                 n_steps=10, n_hidden=512,
                 embedding_dim=512, n_classes=None,
                 learning_rate=0.2, n_to_sample=5000,
                 num_layers=2, embeddings=None,
                 gpu_device="0", full_softmax=False,
                 activation='tanh'):
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.n_hidden = n_hidden
        self.embedding_dim = embedding_dim
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.n_to_sample = n_to_sample
        self.num_layers = num_layers
        self.gpu_device = gpu_device
        self.embeddings = embeddings
        self.full_softmax = full_softmax
        self.activation = activation

        if self.activation == 'tanh':
            self.activation = tf.nn.tanh
        elif self.activation == 'relu':
            self.activation = tf.nn.relu
        elif self.activation == 'sigmoid':
            self.activation = tf.nn.sigmoid

        with tf.device('/gpu:'+str(self.gpu_device)) as dev:
            self.X = [tf.placeholder(tf.int32, [self.batch_size, 1]) for i in range(self.n_steps)]
            self.Y = tf.placeholder(tf.int64, [self.batch_size, 1])
            self.embeddings = tf.Variable(
                tf.random_uniform([self.n_classes,
                                   self.embedding_dim],
                                   -1.0, 1.0))
            self.embed = tf.nn.embedding_lookup(self.embeddings, self.X)
            self.embed = tf.reshape(self.embed, [batch_size, self.embedding_dim*self.n_steps])

            self.hidden_weights = []
            self.hidden_biases = []
            self.hidden_states = []
            if self.num_layers == 1:
                self.hidden_weights.append(tf.Variable(tf.random_normal([self.n_hidden,
                                                                    self.embedding_dim*self.n_steps]),
                                                                    dtype=tf.float32))
                self.hidden_biases.append(tf.Variable(tf.random_normal([self.n_hidden]), dtype=tf.float32))
                self.hidden_states.append(tf.matmul(self.embed, tf.transpose(self.hidden_weights[0]))+self.hidden_biases[0])
            else:
                for i in range(self.num_layers):
                    if i == 0:
                        self.hidden_weights.append(tf.Variable(tf.random_normal([self.n_hidden,
                                                                            self.embedding_dim*self.n_steps]),
                                                                            dtype=tf.float32))
                        self.hidden_biases.append(tf.Variable(tf.random_normal([self.n_hidden]), dtype=tf.float32))
                        self.hidden_states.append(tf.matmul(self.embed, tf.transpose(self.hidden_weights[0]))+self.hidden_biases[0])
                    else:
                        self.hidden_weights.append(tf.Variable(tf.random_normal([self.n_hidden,
                                                                            self.n_hidden]),
                                                                            dtype=tf.float32))
                        self.hidden_biases.append(tf.Variable(tf.random_normal([self.n_hidden]), dtype=tf.float32))
                        self.hidden_states.append(self.activation(tf.matmul(self.hidden_states[i-1], tf.transpose(self.hidden_weights[i]))+self.hidden_biases[i]))


            self.output_weights = tf.Variable(tf.random_normal([self.n_classes,
                                                                self.n_hidden]),
                                                                dtype=tf.float32)

            self.output_biases = tf.Variable(tf.random_normal([self.n_classes]), dtype=tf.float32)            

            self.output_state = tf.matmul(self.hidden_states[-1], tf.transpose(self.output_weights))+self.output_biases



            self.to_s, self.tw, self.ww = tf.nn.learned_unigram_candidate_sampler(self.Y, 1, n_to_sample, False,
                                                                                  self.n_classes, seed=None,
                                                                                  name=None)

            self.full_softmax_loss = nn_ops.sparse_softmax_cross_entropy_with_logits(self.output_state,
                                                                                     tf.reshape(self.Y,
                                                                                     [self.batch_size]))
            self.cost = tf.reduce_mean(self.full_softmax_loss)

            self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate).minimize(self.cost)



    def initialize(self):
        self.saver = tf.train.Saver()
        self.init = tf.initialize_all_variables()
        self.config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=self.config)
        self.sess.run(self.init)