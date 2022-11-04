#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""
from setuptools import find_packages, setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()


with open('requirements/test.txt') as f:
    test_requirements = f.read().splitlines()

with open('README.md') as f:
    long_description = f.read()


setup(
    author='Mitchell Lisle',
    author_email='m.lisle90@gmail.com',
    description='A Python library that adds functionality to asyncio queues',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=requirements,
    include_package_data=True,
    keywords='queueplus',
    name='queueplus',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    test_suite='tests',
    license='MIT license',
    tests_require=test_requirements,
    url='https://github.com/mitchelllisle/queueplus',
    version='0.4.0',
    zip_safe=False,
)
