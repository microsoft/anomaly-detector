# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np


if __name__ == "__main__":
    extensions = [
        Extension(
            "anomaly_detector.univariate._anomaly_kernel_cython",
            ["src/anomaly_detector/univariate/_anomaly_kernel_cython.pyx"],
            include_dirs=[np.get_include()]
        )
    ]

    setup(
        setup_requires=["numpy"],
        ext_modules=cythonize(extensions)
    )
