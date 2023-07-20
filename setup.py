#!/usr/bin/env python

from distutils.core import setup

import re

VERSIONFILE="buddi/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

setup(
    name='BuDDI',
    version=verstr,
    description='BuDDI: Bulk Deconvolution with Domain Invariance',
    author='Natalie R. Davidson',
    author_email='natalie.davidson@cuanschutz.edu',
    python_requires='>=3.7, <4',
    packages=[
        'buddi',
    ],
    install_requires=[
        "anndata==0.8.0",
        "ipywidgets==7.6.5",
        "matplotlib_inline==0.1.6",
        "matplotlib_venn==0.11.6",
        "matplotlib==3.7.1",
        "numpy-groupies==0.9.14",
        "numpy==1.23.5",
        "pandas==1.5.3",
        "pydeseq2==0.3.1",
        "scanpy==1.8.2",
        "scipy==1.8.1",
        "seaborn==0.11.2",
        "tensorflow-estimator==2.12.0",
        "tensorflow-io-gcs-filesystem==0.32.0",
        "tensorflow==2.12.0",
        "tqdm==4.62.3",
        "umap-learn==0.5.2",
        "upsetplot==0.8.0",
    ]
)
