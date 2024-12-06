#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md')) as f:
    long_description = f.read()

setup(
    name='probextreme',
    version='0.0.1',
    description='Python toolbox for Extreme event caracterisation',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/ArcticSnow/probextreme',
    download_url = 'https://github.com/ArcticSnow/probextreme',
    project_urls={
        'Source':'https://github.com/ArcticSnow/probextreme',
    },
    # Author details
    author=['Simon Filhol', 'François Doussot'],
    author_email='simon.filhol@meteo.fr',

    # Choose your license
    license='MIT',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Scientific/Engineering :: Hydrology',
        'Topic :: Scientific/Engineering :: Statistics',
        'Topic :: Scientific/Engineering :: Atmospheric Science',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.11',
    ],

    # What does your project relate to?
    keywords=['climate', 'meteorology', 'extreme'],
    packages=find_packages(),
    python_requires='>=3.10',
    install_requires=['xarray[complete]',
                      'pandas',
                      'matplotlib',
                      'scipy',
                      'numpy==1.26.0',    #  topocalc is having problem with numpy>=2.0.0
                      'netcdf4',
                      'h5netcdf',
                      'pymc',
                      'arviz',
                      'pytensor',
                      'pymc-experimental'],
    include_package_data=True
)
