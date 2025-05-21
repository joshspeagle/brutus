#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re

try:
    from setuptools import setup

except ImportError:
    from distutils.core import setup


dir_path = os.path.dirname(os.path.realpath(__file__))

init_string = open(os.path.join(dir_path, 'brutus', '__init__.py')).read()
VERS = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VERS, init_string, re.M)
__version__ = mo.group(1)

try:
    import pypandoc
    with open('README.md', 'r') as f:
        txt = f.read()
    txt = re.sub('<[^<]+>', '', txt)
    long_description = pypandoc.convert(txt, 'rst', 'md')
except ImportError:
    long_description = open('README.md').read()

    
setup(
    name="astro-brutus",
    url="https://github.com/joshspeagle/brutus",
    version=__version__,
    author="Joshua S Speagle",
    author_email="jspeagle@cfa.harvard.edu",
    packages=["brutus"],
    license="MIT",
    description=("Brute-force Bayesian inference for photometric distances, "
                 "reddenings, and stellar properties"),
    long_description=long_description,
    package_data={"": ["README.md", "LICENSE", "AUTHORS.md"]},
    include_package_data=True,
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "six",
        "h5py",
        "healpy",
        "numba",
        "pooch>=1.4",
    ],
    keywords=["brute force", "photometry", "bayesian", "stellar", "star",
              "cluster", "isochrone", "dust", "reddening",
              "parallax", "distance", "template fitting"],
    classifiers=["Development Status :: 4 - Beta",
                 "License :: OSI Approved :: MIT License",
                 "Natural Language :: English",
                 "Programming Language :: Python :: 2.7",
                 "Programming Language :: Python :: 3.6",
                 "Operating System :: OS Independent",
                 "Topic :: Scientific/Engineering :: Astronomy",
                 "Intended Audience :: Science/Research"]
)
