#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on 2017.08.07 14:52:29

@author: Karim Pedia
"""

# ======================================================================================================================
# ----------------------------------------------------------------------------------------------------- Priority Imports
from __future__ import print_function
import errno
import os
import copy
import os.path as osp
from shutil import copyfile
# -------------------------------------------------------------------------------------------------- Python Lib. Imports
# ----------------------------------------------------------------------------------------------- 3rd Party Lib. Imports
# ---------------------------------------------------------------------------------------------------- Developer Imports
# ------------------------------------------------------------------------------------------------- This package Imports
# ---------------------------------------------------------------------------------------------- This experiment Imports
# ----------------------------------------------------------------------------------------------------------------------
# ======================================================================================================================


# ======================================================================================================================
# ----------------------------------------------------------------------------------------------------------------------
def split_root(path):
    """
    A method that splits a given path into components (root-dir, sub-dir-path).
    For instance: if path='/root/some/path/to/a/file.py', the output value will
    be ('/root', 'some/path/to/a/file.py')
    """
    parent_dir, sub_dir = os.path.split(path)
    if not parent_dir or parent_dir == '/':
        return path, ''
    else:
        rec_parent, ext_parent = split_root(parent_dir)
        return rec_parent, os.path.join(ext_parent, sub_dir)


def makedirs(path):
    """
    A Method that creates a hierarchy of folders from a given path. The path should be a path to a folder because the
      last element will be a directory NOT a file. The method ignores errors, for example if any of the directories do
      exist or so. So, it returns TRUE if the folder is successfully created (or found).
    :param path: [STRING] a path to a directory.
    :return: [BOOLEAN] TRUE if the folder exists, FALSE otherwise.
    """
    if not osp.exists(path):
        try:
            os.makedirs(path)
        except OSError as ose:
            if ose.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else:
                raise
    return osp.exists(path) and osp.isdir(path)


def tmpdir(file_path, run_timestamp=None, class_name=None, sub_dir=None, make=False):
    """
    TODO:
    :param file_path:
    :param run_timestamp:
    :param class_name:
    :param sub_dir:
    :param make:
    :return:
    """
    if file_path is None:
        out_dir = 'tmp_{}'.format(run_timestamp)
        return out_dir if not make else out_dir if osp.isdir(out_dir) else makedirs(out_dir)

    relative_filepath = osp.relpath(file_path, os.getcwd())
    out_dir = os.getcwd()
    out_dir = osp.join(out_dir, 'tmp')
    out_dir = osp.join(out_dir, osp.split(split_root(relative_filepath)[-1])[0])
    out_dir = osp.join(out_dir, osp.splitext(osp.basename(relative_filepath))[0])
    out_dir = osp.join(out_dir, run_timestamp) if run_timestamp is not None else out_dir
    out_dir = osp.join(out_dir, class_name) if class_name is not None else out_dir
    out_dir = osp.join(out_dir, sub_dir) if sub_dir else out_dir

    return out_dir if not make else out_dir if osp.isdir(out_dir) else makedirs(out_dir)


def replaceext(file_list, ext):
    """
    TODO:
    :param file_list:
    :param ext:
    :return:
    """

    out_list = copy.deepcopy(file_list)

    for i in range(len(ext)):
        for j in range(len(out_list)):
            base, extension = osp.splitext(out_list[j])
            out_list[j] = base + (ext[i][1] if extension == ext[i][0] else extension)

    return out_list


def copy_files(file_list, out_dir, fix_ext=(('.pyc', '.py'),)):
    """
    TODO:
    :param file_list:
    :param out_dir:
    :param fix_ext:
    :return:
    """
    rep_list = replaceext(file_list=file_list, ext=fix_ext)

    if not rep_list or not all([osp.exists(x) for x in rep_list]):
        raise IOError(rep_list)

    if not out_dir or not osp.exists(out_dir) or not osp.isdir(out_dir):
        raise IOError(out_dir)

    for x in rep_list:
        copyfile(x, osp.join(out_dir, osp.split(x)[1]))
# ----------------------------------------------------------------------------------------------------------------------
# ======================================================================================================================
