import numpy as np


def split_and_clean(sent):
    sent = [x.strip() for x in sent.split()]
    return sent

def make_feed(train_in, temp_out):
    feed_dict = {}
    i = 0
    for x in X:
        feed_dict[x] = train_in[i]
        i += 1
    i = 0
    feed_dict[Y] = temp_out
    return feed_dict

def test_loss(num_samples=None):
    loss = 0.0
    count = 0
    if num_samples is None:
        num_samples = len(test_in)
    for i in xrange(0, num_samples, 128+10):
        test_grams = np.array([list(x) for x in find_ngrams(test_in[i:i+128+10], 11)], dtype=np.int32)
        if len(test_grams) == 128:
            X_in = test_grams[:, :10].reshape(10, 128, 1)
            Y_in = test_grams[:, 10].reshape(128, 1)
            fd = {}
            for j in range(len(model.X)):
                fd[model.X[j]] = X_in[j]
            fd[model.Y] = Y_in
            l = np.sum(model.sess.run(model.full_softmax_loss, feed_dict=fd))
            loss += l
            count += 128
    return np.exp(loss/float(count)), loss/(float(count))

def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])