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
    mnist = tf_mnist.input_data.read_data_sets('MNIST_data', one_hot=True)
    tfs = tf.InteractiveSession()
    x = tf.placeholder(tf.float32, shape=(None, 784))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    W = tf.Variable(tf.zeros((784, 10)))
    b = tf.Variable(tf.zeros((10,)))

    yh = tf.matmul(x, W) + b
    ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yh))
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(yh, 1)), tf.float32))

    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(ce)

    tfs.run(tf.global_variables_initializer())
    for _ in range(100000):
        batch = mnist.train.next_batch(100)
        train_step.run(feed_dict={x: batch[0], y:batch[1]})
        print('{:05d}) acc = {:0.3f}'.format(_, acc.eval(feed_dict={x:mnist.test.images, y:mnist.test.labels})))
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