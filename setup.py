#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io

from setuptools import setup


def read_file(filename):
    with io.open(filename, encoding='UTF-8') as fp:
        return fp.read().strip()


def read_requirements(filename):
    return [line.strip() for line in read_file(filename).splitlines()
            if not line.startswith('#')]


packages = ['common_cache']

setup(
    name='python-common-cache',
    version=read_file('VERSION'),
    description='This project is an cache component based on the memory and it is lightweight, simple and customizable, you can implement a cache that your needs in a very simple way.',
    long_description=read_file('README.rst'),
    author='SylvanasSun',
    author_email='sylvanas.sun@gmail.com',
    url='https://github.com/SylvanasSun/python-common-cache',
    packages=packages,
    install_requires=read_requirements('requirements.txt'),
    include_package_data=True,
    license='MIT',
    keywords='python cache utils',
    entry_points={
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
