#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys


try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist upload')
    sys.exit()

readme = open('README.md').read()
history = open('HISTORY.md').read().replace('.. :changelog:', '')

setup(
    name='Poubelle',
    version='0.1.0',
    description='Poubelle scripts. Some useful and useless scripts that should be on a poubelle and rewritten.',
    long_description=readme + '\n\n' + history,
    author='Arnaldo Russo',
    author_email='arnaldorusso@gmail.com',
    url='https://github.com/arnaldorusso/Poubelle',
    packages=[
        'Poubelle',
    ],
    package_dir={'Poubelle': 'Poubelle'},
    include_package_data=True,
    install_requires=[
    ],
    license="Python Software License",
    zip_safe=False,
    keywords='Poubelle',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: PSF License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
    ],
    test_suite='tests',
)
