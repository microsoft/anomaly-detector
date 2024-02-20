# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import os
from pathlib import Path
from Cython.Build import cythonize
from setuptools.command.sdist import sdist
from setuptools import find_packages, setup, Extension
import numpy

if __name__ == "__main__":
    import os

    with Path(Path(__file__).parent, "README.md").open(encoding="utf-8") as file:
        long_description = file.read()

    def get_all_package_and_dir(directory):
        all_package = {}
        for path, _, filenames in os.walk(directory):
            if "__init__.py" in filenames and "test" not in path:
                path = "/".join(path.split("\\"))
                package_name = ".".join(path.split("/")[2::])
                all_package[package_name] = path
        return all_package


    def package_files(directory):
        paths = []
        for path, _, filenames in os.walk(directory):
            for filename in filenames:
                paths.append(os.path.join("..", path, filename))
        return paths


    # extra_files = package_files("front/build")

    def _read_reqs(relpath):
        fullpath = os.path.join(os.path.dirname(__file__), relpath)
        with open(fullpath) as f:
            return [
                s.strip()
                for s in f.readlines()
                if (s.strip() and not s.startswith("#"))
            ]

    try:
        REQUIREMENTS = _read_reqs("requirements.txt")
    except Exception:
        REQUIREMENTS = _read_reqs("anomaly_detector.egg-info/requires.txt")

    extensions = [
        Extension("anomaly_detector.univariate._anomaly_kernel_cython",
                  ["anomaly-detector/anomaly_detector/univariate/_anomaly_kernel_cython.pyx"],
                  define_macros=[('CYTHON_TRACE', '1')])
    ]


    # cmdclass = {'build_ext': build_ext}
    # cmdclass.update({'build_ext': build_ext})

    class CustomSdist(sdist):
        def run(self):
            # Run build_ext before sdist
            build_ext_cmd = self.get_finalized_command('build_ext')
            build_ext_cmd.inplace = 1
            self.run_command('build_ext')

            # Use the standard behavior of sdist from the base class
            sdist.run(self)


    cmdclass = {'sdist': CustomSdist}

    all_package = get_all_package_and_dir("./anomaly-detector")
    setup(
        name="anomaly-detector",
        packages=list(all_package.keys()),
        package_dir=all_package,
        ext_modules=cythonize(extensions),
        include_package_data=True,
        cmdclass=cmdclass,
        version="0.1.0",
        license="MIT",
        description="Anomaly Detection",
        long_description=long_description,
        long_description_content_type="text/markdown",
        author="Microsoft",
        author_email="ad-oss@microsoft.com",
        url="https://github.com/microsoft/anomaly-detector",
        data_files=[
            (".", ["README.md"]),
        ],
        keywords=["machine learning", "time series", "anomaly detection"],
        include_dirs=[numpy.get_include()],
        python_requires='>=3.9.0',
        install_requires=REQUIREMENTS,
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
        ],
    )
