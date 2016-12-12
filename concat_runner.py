import os
import json
import numpy as np
from models import concat
from collections import Counter
from utils import *

text = []
with open('data/text8', 'r') as f:
    text = f.read().split()

word_freq = Counter(text)
unique_tokens = word_freq.most_common(50000)
word_index = {unique_tokens[i][0]: i for i in range(len(unique_tokens))}
word_index["<UNK>"] = max(word_index.values()) + 1

batch_size = 128
n_steps = 10
n_hidden = 512
embedding_dim = 512
learning_rate = 0.2
n_classes = len(word_index)

model = concat.ConcatNext(batch_size=batch_size,
                          n_steps=n_steps,
                          n_hidden=n_hidden,
                          embedding_dim=embedding_dim,
                          n_classes=len(word_index),
                          learning_rate=learning_rate,
                          num_layers=4,
                          embeddings=None,
                          gpu_device="1",
                          n_to_sample=100,
                          full_softmax=False,
                          activation='relu')


text = [word_index[y] if word_index.get(y) else word_index['<UNK>'] for y in text]
text = np.array(text, dtype=np.int32)
train_bound = int(len(text) * 0.66)

train_in = text[:train_bound]
test_in = text[train_bound:]
model.initialize()

for e in range(10):
    epoch_loss = 0.0
    c = 0
    for i in xrange(0, len(train_in), 128+10):
        train_grams = np.array([list(x) for x in find_ngrams(train_in[i:i+128+10], 11)], dtype=np.int32)
        X_in = train_grams[:, :10].reshape(10, 128, 1)
        Y_in = train_grams[:, 10].reshape(128, 1)
        fd = {}
        for j in range(len(model.X)):
            fd[model.X[j]] = X_in[j]
        fd[model.Y] = Y_in
        ls, l, _ = model.sess.run([model.full_softmax_loss, model.cost, model.optimizer], feed_dict=fd)
        epoch_loss += np.sum(ls)
        c += len(ls)
        if i % ((128+10)*128) == 0:
            print "EPOCH: " + str(e), float(i)/float(len(train_in))
            print epoch_loss/float(c)
    perp, testloss = test_loss()
    print "PERPLEXITY: " + str(perp)
    print "LOSS: " + str(loss)
    print "--------------------------------------------"




