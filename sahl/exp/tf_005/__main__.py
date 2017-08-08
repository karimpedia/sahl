#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on 2017.08.08 15:09:54

@author: Karim Pedia
"""

# ======================================================================================================================
# ----------------------------------------------------------------------------------------------------- Priority Imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# -------------------------------------------------------------------------------------------------- Python Lib. Imports
import os
import os.path as osp
import urllib
# ----------------------------------------------------------------------------------------------- 3rd Party Lib. Imports
import click
import numpy as np
import tensorflow as tf
# ---------------------------------------------------------------------------------------------------- Developer Imports
# ------------------------------------------------------------------------------------------------- This package Imports
# ---------------------------------------------------------------------------------------------- This experiment Imports
# ----------------------------------------------------------------------------------------------------------------------
# ======================================================================================================================


# ======================================================================================================================
# ----------------------------------------------------------------------------------------------------------------------
SYS_ROOT = osp.abspath(os.sep)
IRIS_TRAINING, IRIS_TEST = \
    osp.join(SYS_ROOT,'tmp','sahl','iris','iris_training.csv'), \
    osp.join(SYS_ROOT,'tmp','sahl','iris','iris_test.csv')
IRIS_TRAINING_URL, IRIS_TEST_URL = \
    'http://download.tensorflow.org/data/iris_training.csv', \
    'http://download.tensorflow.org/data/iris_test.csv'
TMP_MODEL_DIR = osp.join(SYS_ROOT,'tmp','sahl','iris')

# ----------------------------------------------------------------------------------------------------------------------
# ======================================================================================================================


# ======================================================================================================================
# ----------------------------------------------------------------------------------------------------------------------
def download_dataset():
    if not osp.exists(IRIS_TRAINING):
        tf.gfile.MakeDirs(osp.split(IRIS_TRAINING)[0])
        raw = urllib.urlopen(IRIS_TRAINING_URL).read()
        with open(IRIS_TRAINING, 'w') as x:
            x.write(raw)

    if not osp.exists(IRIS_TEST):
        tf.gfile.MakeDirs(osp.split(IRIS_TEST)[0])
        raw = urllib.urlopen(IRIS_TRAINING_URL).read()
        with open(IRIS_TEST, 'w') as x:
            x.write(raw)


def load_dataset():
    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename=IRIS_TRAINING,
                                                                       target_dtype=np.int,
                                                                       features_dtype=np.float32)
    test_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename=IRIS_TEST,
                                                                   target_dtype=np.int,
                                                                   features_dtype=np.float32)
    return training_set, test_set


def main():
    download_dataset()
    trnD, vldD = load_dataset()

    def get_train_inputs(): return tf.constant(trnD.data), tf.constant(trnD.target)
    def get_test_inputs(): return tf.constant(vldD.data), tf.constant(vldD.target)
    def get_new_inputs(): return tf.constant([[6.4,3.2,4.5,1.5],[5.8,3.1,5.0,1.7]], dtype=tf.float32)

    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]
    print(type(feature_columns[0]))

    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                                hidden_units=[10, 20 , 10],
                                                n_classes=3,
                                                model_dir=TMP_MODEL_DIR)
    classifier.fit(input_fn=get_train_inputs, steps=3)
    acc = classifier.evaluate(input_fn=get_test_inputs, steps=1)['accuracy']
    print('Test accuracy: {0:.5f}'.format(acc))
    prd = list(classifier.predit(input_fn=get_new_inputs))
    print('New class predictions: {}'.format(prd))


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

click.CommandCollection(sources=[cmd])()# ----------------------------------------------------------------------------------------------------------------------
# ======================================================================================================================
