from unicodedata import name
from setuptools import setup

with open("README.md","r") as fh:
    long_description = fh.read()

setup(
    name = 'dawsonmuse',
    version = '0.0.1',
    description= 'A module for running EEG experiments with Psychopy and a Muse device.',
    py_modules=["dawsonmuse"],
    author="Tokiniaina Raharison Ralambomihanta",
    author_email="raharisonrtoky@gmail.com",
    url="https://github.com/alexandrebarachant/muse-lsl/",
    package_dir={'':'src'},
    classifiers=[
        "Programming Language :: Python ::3",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: BSD License"
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "muselsl",
        "mne",
    ],
)