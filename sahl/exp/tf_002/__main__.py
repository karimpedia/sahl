#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on 2017.08.07 15:43:50

@author: Karim Pedia
"""

# ======================================================================================================================
# ----------------------------------------------------------------------------------------------------- Priority Imports
from __future__ import print_function
# -------------------------------------------------------------------------------------------------- Python Lib. Imports
# ----------------------------------------------------------------------------------------------- 3rd Party Lib. Imports
import click
import numpy as np
import tensorflow as tf
import tensorflow.examples.tutorials.mnist as tf_mnist
# ---------------------------------------------------------------------------------------------------- Developer Imports
# ------------------------------------------------------------------------------------------------- This package Imports
# ---------------------------------------------------------------------------------------------- This experiment Imports
from sahl.util.tf_util import get_tf_session
# ----------------------------------------------------------------------------------------------------------------------
# ======================================================================================================================


# ======================================================================================================================
# ----------------------------------------------------------------------------------------------------------------------
# DECLARE CONSTANTS HERE
# ----------------------------------------------------------------------------------------------------------------------
# ======================================================================================================================


# ======================================================================================================================
# ----------------------------------------------------------------------------------------------------------------------
def main():
    mnist = tf_mnist.input_data.read_data_sets('/tmp/tensorflow/mnist/input_data', one_hot=True)

    print(np.min(mnist.test.images), np.max(mnist.test.images))

    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)

    W = [tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1)),
         tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1)),
         tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1)),
         tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))]
    b = [tf.Variable(tf.constant(0.1, shape=[32])),
         tf.Variable(tf.constant(0.1, shape=[64])),
         tf.Variable(tf.constant(0.1, shape=[1024])),
         tf.Variable(tf.constant(0.1, shape=[10]))]

    xt = tf.reshape(x, [-1, 28, 28, 1])

    xt = tf.nn.relu(tf.nn.conv2d(xt, W[0], strides=[1, 1, 1, 1], padding='SAME') + b[0])
    xt = tf.nn.max_pool(xt, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    xt = tf.nn.relu(tf.nn.conv2d(xt, W[1], strides=[1, 1, 1, 1], padding='SAME') + b[1])
    xt = tf.nn.max_pool(xt, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    xt = tf.reshape(xt, [-1, 7 * 7 * 64])
    xt = tf.nn.relu(tf.matmul(xt, W[2]) + b[2])
    xt = tf.nn.dropout(xt, keep_prob)
    xt = tf.matmul(xt, W[3]) + b[3]

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=xt))
    training = tf.train.AdamOptimizer(1e-4).minimize(loss)
    predictions = tf.equal(tf.argmax(xt, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(predictions, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(20000):
            batch = mnist.train.next_batch(50)
            training.run(feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
                test_accuracy = accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
                print('step %d, training accuracy: %g, test accuracy: %g' % (i, train_accuracy, test_accuracy))
        print(
            'test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}))




        # Should reach ~92% of test accuracy
# ----------------------------------------------------------------------------------------------------------------------
# ======================================================================================================================


# ======================================================================================================================
# ----------------------------------------------------------------------------------------------------------------------
@click.group()
def cmd():
    pass

@cmd.command()
def cmd_main():
    main()

click.CommandCollection(sources=[cmd])()
# ----------------------------------------------------------------------------------------------------------------------
# ======================================================================================================================
