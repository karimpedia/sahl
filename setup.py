#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on 2017.08.03 03:41:31 PM

@author: karim pedia
"""

from setuptools import setup
from setuptools import find_packages


setup_options = dict(name='sahl',
                     version='0.0.1',
                     author='Karim Pedia',
                     author_email='karim.pedia@gmail.com',
                     packages=['sahl'],
                     install_requires=[],
                     extras_require={'visualize': [],
                                     'tests': [],
                                     "docs": []})

if __name__ == '__main__':
    setup(**setup_options)
