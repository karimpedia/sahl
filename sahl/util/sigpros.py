#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 10:52:09 2017

@author: karim
"""

# =================================================================================================
import numpy as np
# .................................................................................................
from ukdale.util.np_util import unset_short_sets
from ukdale.util.timestamp import TS
# =================================================================================================


def threshold_based_gt(signal, power_threshold, min_on_duration, min_off_duration, verbose=False):
    if verbose:
        print(TS('Constructing ground truth signals'))
    signal = (signal > power_threshold).astype(np.float32)
    signal = 1.0 - unset_short_sets(1.0 - signal, min_off_duration)
    return unset_short_sets(signal, min_on_duration)


def on_periods(fOnSignal):
    tempDiff = np.diff(np.pad(fOnSignal.astype(np.float32), 1, 'constant', constant_values=(0, 0)))
    return zip(np.where(tempDiff > 0)[0], np.where(tempDiff < 0)[0])
