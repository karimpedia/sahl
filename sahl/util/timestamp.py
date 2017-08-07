#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on 2017.08.07 14:52:29

@author: Karim Pedia
"""

# ======================================================================================================================
# ----------------------------------------------------------------------------------------------------- Priority Imports
from __future__ import print_function
from datetime import datetime
from types import StringTypes
# -------------------------------------------------------------------------------------------------- Python Lib. Imports
# ----------------------------------------------------------------------------------------------- 3rd Party Lib. Imports
# ---------------------------------------------------------------------------------------------------- Developer Imports
# ------------------------------------------------------------------------------------------------- This package Imports
# ---------------------------------------------------------------------------------------------- This experiment Imports
# ----------------------------------------------------------------------------------------------------------------------
# ======================================================================================================================


# ======================================================================================================================
# ----------------------------------------------------------------------------------------------------------------------
def tsprint(msg=None, dots=True, ignore_newlines=True):
    timestamp = datetime.now().strftime('%Y.%m.%d %H:%M:%S.%f')
    prefix = '{}) '.format(timestamp)
    nxtfix = '{}) '.format('.' * len(timestamp)) if dots else prefix
    if msg is None:
        return prefix
    else:
        msg = [msg] if isinstance(msg, StringTypes) else msg
        if not ignore_newlines:
            msg = [i for j in [x.split('\n') for y in msg for x in y.split('\r\n')] for i in j]
        msg = ['{}{}'.format(prefix, msg[0])] + ['{}{}'.format(nxtfix, x) for x in msg[1:]]
        msg = '\r\n'.join(msg)
        return msg
# ----------------------------------------------------------------------------------------------------------------------
# ======================================================================================================================
