#!/usr/bin/env python

import imp
from setuptools import setup, find_packages


setup(
    name='soccer',
    version='0.1.0',
    packages=find_packages(),
    install_requires=['pyglet', 'gym'],
)
