#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 16:10:19 2017

@author: ks
"""


# ======================================================================================================================
# ----------------------------------------------------------------------------------------------------- Priority Imports
# -------------------------------------------------------------------------------------------------- Python Lib. Imports
import subprocess
# ----------------------------------------------------------------------------------------------- 3rd Party Lib. Imports
# ---------------------------------------------------------------------------------------------------- Developer Imports
# ------------------------------------------------------------------------------------------------- This package Imports
# ---------------------------------------------------------------------------------------------- This experiment Imports
# ----------------------------------------------------------------------------------------------------------------------
# ======================================================================================================================


# ======================================================================================================================
# ----------------------------------------------------------------------------------------------------------------------
def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'])


def get_git_revision_short_hash():
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])
# ----------------------------------------------------------------------------------------------------------------------
# ======================================================================================================================