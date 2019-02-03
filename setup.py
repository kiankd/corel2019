# Always prefer setuptools over distutils
from setuptools import setup, find_packages

# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='corel2019',
    version='0.0.1',
    description='Clustering-oriented representation learning with Attractive-Repulsive loss',
    long_description=long_description,
    url='https://github.com/kiankd/corel2019',
    author='Kian Kenyon-Dean',
    author_email='kiankd@gmail.com',
    license='GNU General Public License v3.0',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
    keywords= (
        'clustering, deep learning, pytorch wrappers'
    ),
    packages=['corel'],
	package_data={'': ['README.md']},
	install_requires=[
        'numpy'
    ]
)
