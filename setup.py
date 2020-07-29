from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='lvmwrappers',
    version='0.2',
    url='https://github.com/goncalomcorreia/explicit-sparse-marginalization',
    author='Gonçalo Correia',
    author_email='goncalommac@gmail.com',
    packages=find_packages(),
    install_requires=requirements,
)
