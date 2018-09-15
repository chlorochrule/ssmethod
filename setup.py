# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='ssmethod',
    version='0.1.0',
    description='',
    long_description=readme,
    author='Naoto MINAMI',
    author_email='minami.polly@gmail.com',
    install_requires=['numpy', 'scipy', 'sklearn'],
    url='https://github.com/chlorochrule/ssmethod',
    license=license,
    packages=find_packages(exclude=('tests', 'examples'))
)
