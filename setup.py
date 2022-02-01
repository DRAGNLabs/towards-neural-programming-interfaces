from setuptools import find_packages, setup

setup(
    name='dragn.npi',
    packages=find_packages("src"),
    package_dir={'': 'src'},
    version='0.1.0',
    description='Official implementation of "Towards Neural Programming Interfaces"',
    author='DRAGN',
    license='Apache 2.0',
)
