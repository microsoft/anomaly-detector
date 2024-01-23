import os
from pathlib import Path
from Cython.Build import cythonize
from setuptools.command.build_ext import build_ext
from setuptools.command.sdist import sdist
from setuptools import find_packages, setup, Extension
import numpy

if __name__ == "__main__":
    import os


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


    REQUIREMENTS = _read_reqs("requirements.txt")

    extensions = [
    Extension("anomaly_detector.univariate._anomaly_kernel_cython", ["anomaly-detector/anomaly_detector/univariate/_anomaly_kernel_cython.pyx"],
              define_macros=[('CYTHON_TRACE', '1')])
]
    # cmdclass = {'build_ext': build_ext}
    # cmdclass.update({'build_ext': build_ext})

    class CustomSdist(sdist):
        def run(self):
            # Run build_ext before sdist
            self.run_command('build_ext')
            build_ext_cmd = self.get_finalized_command('build_ext')
            build_ext_cmd.inplace = 1

            # Use the standard behavior of sdist from the base class
            sdist.run(self)
    
    cmdclass = {'sdist': CustomSdist}

    setup(
        name="anomaly_detector",
        packages=["anomaly_detector", "anomaly_detector.common", "anomaly_detector.multivariate", "anomaly_detector.univariate"],
        package_dir={
            "anomaly_detector": "./anomaly-detector/anomaly_detector",
            "anomaly_detector.common": "./anomaly-detector/anomaly_detector/common",
            "anomaly_detector.multivariate": "./anomaly-detector/anomaly_detector/multivariate",
            "anomaly_detector.univariate": "./anomaly-detector/anomaly_detector/univariate",
        },
        ext_modules=cythonize(extensions),
        include_package_data=True,
        cmdclass=cmdclass,
        version="0.1.0",
        license="MIT",
        description="Anomaly Detection",
        # long_description=long_description,
        # long_description_content_type="text/markdown",
        # entry_points={"console_scripts": [""]},
        author="test",
        author_email="anomaly_detector@microsoft.com",
        url="https://github.com/microsoft/anomaly-detector",
        data_files=[
            (".", ["README.md"]),
        ],
        keywords=["machine learning", "time series", "anomaly detection"],
        include_dirs=[numpy.get_include()],
        python_requires='>=3.8.0',
        install_requires=REQUIREMENTS,
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3.11",
        ],
    )
