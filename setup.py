# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

d = {}
exec(open("neuroplotlib/version.py").read(), None, d)
version = d['version']
long_description = open("README.md").read()

entry_points = None

install_requires = []

setup(
    name="neuroplotlib",
    version=version,
    author="Alessio Buccino",
    author_email="alessiop.buccino@gmail.com",
    description="Python package for visualization of neurons in LFPy.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alejoe91/neuroplotlib",
    install_requires=[
        'numpy',
        'matplotlib',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    entry_points=entry_points,
    include_package_data=True,
)
