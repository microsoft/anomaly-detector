# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from setuptools import setup, Extension
from Cython.Build import cythonize

class GetNumpyInclude(object):
    """A lazy include path for numpy.
    This way numpy isn't imported until it's actually installed,
    so the `install_requires` argument can handle it properly.
    """
    def __str__(self):
        import numpy
        return numpy.get_include()


if __name__ == "__main__":
    extensions = [
        Extension(
            "anomaly_detector.univariate._anomaly_kernel_cython",
            ["src/anomaly_detector/univariate/_anomaly_kernel_cython.pyx"],
            include_dirs=[GetNumpyInclude()]
        )
    ]

    setup(
        setup_requires=["numpy"],
        ext_modules=cythonize(extensions)
    )
