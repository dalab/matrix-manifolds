import os
import re
from setuptools import find_packages, setup, Extension

from Cython.Build import cythonize
import numpy as np

# NOTE: This should be installed outside conda environments and on a system with
# GCC >=9.  To do it from within a conda environment, without exiting it, use:
#
#       env -i bash -l -c '/usr/bin/python setup.py develop --user'

DESCRIPTION = """Learning representations in matrix manifolds."""
PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))


def get_version(*path):
    version_file = os.path.join(*path)
    lines = open(version_file, 'rt').readlines()
    version_regex = r"^__version__ = ['\"]([^'\"]*)['\"]"
    for line in lines:
        mo = re.search(version_regex, line, re.M)
        if mo:
            return mo.group(1)
    raise RuntimeError('Unable to find version in {}.'.format(version_file))


setup(
        name='graphembed',
        author='Calin Cruceru',
        author_email='ccruceru@ethz.ch',
        description=DESCRIPTION,
        version=get_version(PROJECT_ROOT, 'graphembed', '__init__.py'),
        packages=find_packages(),
        ext_modules=cythonize([
                Extension(
                        'graphembed.pyx.precision',
                        sources=['graphembed/pyx/precision.pyx'],
                        include_dirs=[np.get_include()],
                        extra_compile_args=['-std=c++17', '-I.'],
                        extra_link_args=['-ltbb'],
                ),
        ]),
        python_requires='>=3.7',
        install_requires=['cython', 'numpy'],
        tests_require=['pytest'],
        test_suite='tests',
        license='BSD 3-Clause License',
)
