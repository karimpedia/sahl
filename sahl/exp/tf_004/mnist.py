#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on 2017.08.08 11:45:16

@author: Karim Pedia
"""

# ======================================================================================================================
# ----------------------------------------------------------------------------------------------------- Priority Imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# -------------------------------------------------------------------------------------------------- Python Lib. Imports
import math
# ----------------------------------------------------------------------------------------------- 3rd Party Lib. Imports
import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------------------------------------- Developer Imports
# ------------------------------------------------------------------------------------------------- This package Imports
# ---------------------------------------------------------------------------------------------- This experiment Imports
# ----------------------------------------------------------------------------------------------------------------------
# ======================================================================================================================


# ======================================================================================================================
# ----------------------------------------------------------------------------------------------------------------------
NUM_CLASSES = 10
IMAGE_PIXELS = 28 * 28
# ----------------------------------------------------------------------------------------------------------------------
# ======================================================================================================================


# ======================================================================================================================
# ----------------------------------------------------------------------------------------------------------------------
def inference(images, hidden=(300, 100)):
    depth = -1

    depth += 1
    fanin = IMAGE_PIXELS
    datain = images
    with tf.name_scope('H{:03d}'.format(depth)):
        weights = tf.Variable(tf.truncated_normal([fanin, hidden[depth]], stddev=1.0/math.sqrt(float(fanin))), name='weights')
        biases = tf.Variable(tf.zeros([hidden[depth]]), name='biases')
        dataout = tf.nn.relu(tf.matmul(datain, weights) + biases)

    depth += 1
    fanin = hidden[depth-1]
    datain = dataout
    with tf.name_scope('H{:03d}'.format(depth)):
        weights = tf.Variable(tf.truncated_normal([fanin, hidden[depth]], stddev=1.0/math.sqrt(float(fanin))), name='weights')
        biases = tf.Variable(tf.zeros([hidden[depth]]), name='biases')
        dataout = tf.nn.relu(tf.matmul(datain, weights) + biases)

    depth += 1
    fanin = hidden[depth-1]
    datain = dataout
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(tf.truncated_normal([fanin, NUM_CLASSES], stddev=1.0/math.sqrt(float(fanin))), name='weights')
        biases = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases')
        dataout = tf.matmul(datain, weights) + biases

    return dataout


def loss(logits, labels):
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='xentropy')
    return tf.reduce_mean(cross_entropy, name='xentropy_mean')


def training(loss, learning_rate):
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    training_op = optimizer.minimize(loss, global_step=global_step)
    return training_op

def evaluation(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, k=1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))


# ----------------------------------------------------------------------------------------------------------------------
# ======================================================================================================================


# ======================================================================================================================
# ----------------------------------------------------------------------------------------------------------------------
# CMD TOOLS SHOULD BE HERE
# ----------------------------------------------------------------------------------------------------------------------
# ======================================================================================================================
