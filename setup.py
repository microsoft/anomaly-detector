# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from setuptools import setup, Extension
import numpy as np

if __name__ == "__main__":
    setup(
        include_dirs=[np.get_include()],
        ext_modules=[
            Extension(
                "anomaly_detector.univariate._anomaly_kernel_cython",
                ["src/anomaly_detector/univariate/_anomaly_kernel_cython.c"],
            )
        ]
    )
