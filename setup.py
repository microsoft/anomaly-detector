# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from setuptools import setup, Extension

class get_numpy_include(object):
    """A lazy include path for numpy.
    This way numpy isn't imported until it's actually installed,
    so the `install_requires` argument can handle it properly.
    """
    def __str__(self):
        import numpy
        return numpy.get_include()


if __name__ == "__main__":
    setup(
        include_dirs=[get_numpy_include()],
        ext_modules=[
            Extension(
                "anomaly_detector.univariate._anomaly_kernel_cython",
                ["src/anomaly_detector/univariate/_anomaly_kernel_cython.c"],
            )
        ]
    )
