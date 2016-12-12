import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
from tensorflow.python.ops import nn_ops

class LSTM():

    def __init__(self, batch_size=128,
                 n_steps=10, n_hidden=512,
                 embedding_dim=512, n_classes=None,
                 learning_rate=0.2, n_to_sample=5000,
                 num_lstm_layers=2, embeddings=None,
                 gpu_device="0", full_softmax=False):
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.n_hidden = n_hidden
        self.embedding_dim = embedding_dim
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.n_to_sample = n_to_sample
        self.num_lstm_layers = num_lstm_layers
        self.gpu_device = gpu_device
        self.embeddings = embeddings
        self.full_softmax = full_softmax

        with tf.device('/gpu:'+str(self.gpu_device)) as dev:
            self.X = [tf.placeholder(tf.int32, [self.batch_size, 1]) for i in range(self.n_steps)]
            self.Y = tf.placeholder(tf.int64, [self.batch_size, 1])
            self.weights = tf.Variable(tf.random_normal([self.n_hidden, self.n_classes]), dtype=tf.float32)
            self.biases = tf.Variable(tf.random_normal([self.n_classes]), dtype=tf.float32)
            self.lstm_cell = rnn_cell.DropoutWrapper(rnn_cell.BasicLSTMCell(self.n_hidden, forget_bias=1.0,
                                                                            state_is_tuple=True),
                                                                            output_keep_prob=0.5)
            self.stacked_lstm = rnn_cell.MultiRNNCell([self.lstm_cell] * self.num_lstm_layers,
                                                       state_is_tuple=True)
            self.lstm_embed_cell = rnn_cell.EmbeddingWrapper(self.stacked_lstm,
                                                             embedding_classes=self.n_classes,
                                                             embedding_size=embedding_dim)
            self.outputs, self.state = rnn.rnn(self.lstm_embed_cell, self.X, dtype=tf.float32)
            self.to_s, self.tw, self.ww = tf.nn.learned_unigram_candidate_sampler(self.Y, 1, n_to_sample, False,
                                                                                  self.n_classes, seed=None,
                                                                                  name=None)
            self.loss = tf.nn.sampled_softmax_loss(tf.transpose(self.weights),
                                                   self.biases, self.outputs[-1], self.Y, n_to_sample,
                                                   self.n_classes, num_true=1,
                                                   sampled_values=(self.to_s, self.tw, self.ww),
                                                   remove_accidental_hits=True,
                                                   partition_strategy='mod',
                                                   name='sampled_softmax_loss')
            self.full_softmax_loss = nn_ops.sparse_softmax_cross_entropy_with_logits(tf.matmul(self.outputs[-1],
                                                                                           self.weights) + self.biases,
                                                                                          tf.reshape(self.Y,
                                                                                          [self.batch_size]))
            if self.full_softmax:
                self.cost = tf.reduce_mean(self.full_softmax_loss)
            else:
                self.cost = tf.reduce_mean(self.loss)

            self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate).minimize(self.cost)



    def initialize(self):
        self.saver = tf.train.Saver()
        self.init = tf.initialize_all_variables()
        self.config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=self.config)
        self.sess.run(self.init)