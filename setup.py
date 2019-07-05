from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='dynamicimage',
    version='0.1',
    install_requires=requirements,
    packages=['dynamicimage'],
    url='',
    license='MIT',
    author='Rick Wu',
    author_email='',
    description=''
)
