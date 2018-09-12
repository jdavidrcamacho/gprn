#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from setuptools import setup

setup(name='gprn',
      version='0.1',
      description='Implementation of a thing with GPs',
      author='João Camacho',
      author_email='joao.camacho@astro.up.pt',
      license='MIT',
      url='https://github.com/jdavidrcamacho/Tests_gprn',
      packages=['gprn'],
      install_requires=[
        'numpy',
        'scipy'
      ],
     )
