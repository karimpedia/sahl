#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on 2017.08.08 12:46:26

@author: Karim Pedia
"""

# ======================================================================================================================
# ----------------------------------------------------------------------------------------------------- Priority Imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# -------------------------------------------------------------------------------------------------- Python Lib. Imports
import time
import os.path as osp
# ----------------------------------------------------------------------------------------------- 3rd Party Lib. Imports
import click
import tensorflow as tf
import tensorflow.examples.tutorials.mnist as tf_mnist
# ---------------------------------------------------------------------------------------------------- Developer Imports
# ------------------------------------------------------------------------------------------------- This package Imports
# ---------------------------------------------------------------------------------------------- This experiment Imports
from . import mnist
# ----------------------------------------------------------------------------------------------------------------------
# ======================================================================================================================


# ======================================================================================================================
# ----------------------------------------------------------------------------------------------------------------------
# DECLARE CONSTANTS HERE
# ----------------------------------------------------------------------------------------------------------------------
# ======================================================================================================================


# ======================================================================================================================
# ----------------------------------------------------------------------------------------------------------------------
def main(learning_rate=0.1, num_steps=20000, batch_size=100, hidden_layers=(300, 100),
         input_data_dir='/tmp/tensorflow/mnist/input_data',
         log_dir='/tmp/tensorflow/mnist/logs/fully_connected_feed'):

    tf.gfile.DeleteRecursively(log_dir) if tf.gfile.Exists(log_dir) else None
    tf.gfile.MakeDirs(log_dir)

    datasets = tf_mnist.input_data.read_data_sets(input_data_dir)

    with tf.Graph().as_default():
        images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, mnist.IMAGE_PIXELS))
        labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size,))
        logits = mnist.inference(images_placeholder, hidden=hidden_layers)
        loss = mnist.loss(logits=logits, labels=labels_placeholder)
        train_op = mnist.training(loss=loss, learning_rate=learning_rate)
        eval_correct = mnist.evaluation(logits=logits, labels=labels_placeholder)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        # FIXME: summary = tf.summary.merge_all()

        sess = tf.Session()

        # FIXME: summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        sess.run(init)
        for step in range(num_steps):
            start_time = time.time()
            next_batch = datasets.train.next_batch(batch_size)
            feed_dict = {images_placeholder: next_batch[0], labels_placeholder: next_batch[1]}
            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
            duration = time.time() - start_time
            if step % 100 == 0:
                print('Step {step:0{numdigits:d}d}/{num_steps} [{duration:.2f}S]) cst: {loss}'
                      .format(numdigits=len(str(num_steps)), num_steps=num_steps, step=step, duration=duration, loss=loss_value))
                # FIXME: summary_str = sess.run(summary, feed_dict=feed_dict)
                # FIXME: summary_writer.add_summary(summary_str, step)
                # FIXME: summary_writer.flush()
            if (step + 1) % 1000 == 0 or (step + 1) == num_steps:
                checkpoint_file = osp.join(log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step)
                eval(sess, eval_correct, images_placeholder, labels_placeholder, datasets.train, batch_size, 'trn)')
                eval(sess, eval_correct, images_placeholder, labels_placeholder, datasets.validation, batch_size, 'vld)')
                eval(sess, eval_correct, images_placeholder, labels_placeholder, datasets.test, batch_size, 'evl)')


def eval(sess, eval_correct, images_placeholder, labels_placehoder, dataset, batch_size, header_msg):
    true_count = 0
    steps_per_epoch = dataset.num_examples // batch_size
    num_examples = steps_per_epoch * batch_size
    for steps in range(steps_per_epoch):
        next_batch = dataset.next_batch(batch_size)
        feed_dict = {images_placeholder: next_batch[0], labels_placehoder: next_batch[1]}
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    accuracy = true_count / num_examples
    print(header_msg + '#Samples: {:05d}, #True: {:05d}, RAC: {:0.5f}'.format(num_examples, true_count, accuracy))
# ----------------------------------------------------------------------------------------------------------------------
# ======================================================================================================================


# ======================================================================================================================
# ----------------------------------------------------------------------------------------------------------------------
@click.group()
def cmd():
    pass

@cmd.command()
def cmain():
    main()

click.CommandCollection(sources=[cmd])()
# ----------------------------------------------------------------------------------------------------------------------
# ======================================================================================================================
