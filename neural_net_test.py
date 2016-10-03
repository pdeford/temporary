#!/usr/bin/env python

import edward as ed
import numpy as np
import tensorflow as tf
import tables

from edward.models import Normal
from edward.stats import norm
from sklearn.linear_model import LogisticRegression as logit
from sklearn.metrics import roc_auc_score, average_precision_score


def neural_network(x, W_0, W_1, b_0, b_1):
    h = tf.nn.tanh(tf.matmul(x, W_0) + b_0)
    h = tf.matmul(h, W_1) + b_1
    return tf.reshape(h, [-1])

# DATA
print "LOADING DATA"
name = "ATF2_down.h5"

f = tables.open_file(name, mode='r')
y_train = f.root.Y[:]
x_train = f.root.X[:]
f.close()

x_place = tf.placeholder(tf.float32, shape=x_train.shape)

N = x_train.shape[0]   # num data ponts
D = x_train.shape[1]   # num features

# MODEL
print "DEFINING MODEL"
W_0 = Normal(mu=tf.zeros([D, 2]), sigma=tf.ones([D, 2]))
W_1 = Normal(mu=tf.zeros([2, 1]), sigma=tf.ones([2, 1]))
b_0 = Normal(mu=tf.zeros(2), sigma=tf.ones(2))
b_1 = Normal(mu=tf.zeros(1), sigma=tf.ones(1))

x = tf.convert_to_tensor(tf.Variable(x_place), dtype=tf.float32)
y = Normal(mu=neural_network(x, W_0, W_1, b_0, b_1),
           sigma=0.1 * tf.ones(N))

# INFERENCE
print "PREPARING TO CARRY OUT INFERENCE"
qW_0 = Normal(mu=tf.Variable(tf.random_normal([D, 2])),
              sigma=tf.nn.softplus(tf.Variable(tf.random_normal([D, 2]))))
qW_1 = Normal(mu=tf.Variable(tf.random_normal([2, 1])),
              sigma=tf.nn.softplus(tf.Variable(tf.random_normal([2, 1]))))
qb_0 = Normal(mu=tf.Variable(tf.random_normal([2])),
              sigma=tf.nn.softplus(tf.Variable(tf.random_normal([2]))))
qb_1 = Normal(mu=tf.Variable(tf.random_normal([1])),
              sigma=tf.nn.softplus(tf.Variable(tf.random_normal([1]))))

data = {y: y_train, x_place: x_train}
inference = ed.MFVI({W_0: qW_0, b_0: qb_0,
                     W_1: qW_1, b_1: qb_1}, data)


sess = ed.get_session()
init = tf.initialize_all_variables()
init.run(feed_dict={x_place: x_train})

# RUN MEAN-FIELD VARIATIONAL INFERENCE
print "INFERENCE"
inference.run(n_iter=500, n_samples=5, n_print=100,)

# GET FITS, AND LEARN LOGISTIC REGRESSION MODEL ON OUTPUT
print "TRAIN LOGIT"
mus = neural_network(x_train, qW_0.sample(), qW_1.sample(),
                     qb_0.sample(), qb_1.sample())
outputs = mus.eval()
outputs = outputs.reshape(-1,1)

clf = logit()
clf.fit(outputs, y_train)

# SCORE THE PERFORMANCE OF THE FULL MODEL
print "EVAL PERFORMANCE"

Y2 = clf.predict_proba(outputs)[:,1]
score1 = average_precision_score(y_train, Y2)
score2 = roc_auc_score(y_train, Y2)

print "auPRC:", score1
print "auROC:", score2
