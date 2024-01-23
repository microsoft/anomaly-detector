import os
import os
import numpy as np
from distutils.errors import CCompilerError, DistutilsExecError, DistutilsPlatformError

# See if Cython is installed
try:
    from Cython.Build import cythonize
# Do nothing if Cython is not available
except ImportError:
    # Got to provide this function. Otherwise, poetry will fail
    def build(setup_kwargs):
        pass
# Cython is installed. Compile
else:
    from setuptools import Extension
    from setuptools.dist import Distribution
    from setuptools.command.build_ext import build_ext
# use cythonize to build the extensions
modules = ["./anomaly_detector/univariate/*.pyx"]
extensions = cythonize(modules,
                       language_level=3,
                       compiler_directives={'linetrace': True},
                       )


# cmdclass = {'build_ext': build_ext}
class BuildFailed(Exception):
    pass


class ExtBuilder(build_ext):

    def run(self):
        try:
            build_ext.run(self)
        except (DistutilsPlatformError, FileNotFoundError):
            raise BuildFailed('File not found. Could not compile C extension.')

    def build_extension(self, ext):
        try:
            build_ext.build_extension(self, ext)
        except (CCompilerError, DistutilsExecError, DistutilsPlatformError, ValueError):
            raise BuildFailed('Could not compile C extension.')


cmdclass = {"build_ext": ExtBuilder}


def build(setup_kwargs):
    """Needed for the poetry building interface."""

    os.environ['CFLAGS'] = '-O3'

    setup_kwargs.update({
        'ext_modules': extensions,
        'include_dirs': [np.get_include()],
        'cmdclass': cmdclass
    })
