import numpy as np
import os
import sys
from copy import deepcopy
from os.path import join as pjoin, dirname, exists
from glob import glob

# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if exists('MANIFEST'): os.remove('MANIFEST')

# force_setuptools can be set from the setup_egg.py script
if not 'force_setuptools' in globals():
    # For some commands, always use setuptools
    if len(set(('develop', 'bdist_egg', 'bdist_rpm', 'bdist', 'bdist_dumb',
                'bdist_mpkg', 'bdist_wheel', 'install_egg_info', 'egg_info',
                'easy_install')).intersection(sys.argv)) > 0:
        force_setuptools = True
    else:
        force_setuptools = False

if force_setuptools:
    import setuptools

# Import distutils _after_ potential setuptools import above, and after removing
# MANIFEST
from distutils.core import setup
from distutils.extension import Extension
from distutils.command.build_py import build_py as du_build_py
from distutils.command.build_ext import build_ext as du_build_ext

# from cythexts import cyproc_exts, get_pyx_sdist, derror_maker
# from setup_helpers import (install_scripts_bat, add_flag_checking,
                           # SetupDependency, read_vars_from)

def main():
    setup(
          packages     = ['core',
                          'core.objects'
                          ],
        )


#simple way to test what setup will do
#python setup.py install --prefix=/tmp
if __name__ == "__main__":
    main()
