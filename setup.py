import os
from pathlib import Path
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from setuptools import find_packages, setup,Extension
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
    cmdclass = {'build_ext': build_ext}
    cmdclass.update({'build_ext': build_ext})

    setup(
        name="anomaly_detector",
        packages=['anomaly_detector'],
        package_dir={"anomaly_detector": "./anomaly-detector/anomaly_detector"},
        # package_data={"": ['*.txt']},
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
        author_email="juaduan@microsoft.com",
        url="https://github.com/juaduan/anomaly-detector",
        data_files=[
            (".", ["README.md"]),
        ],
        keywords=["machine learning", "time series", "anomaly detection"],
        include_dirs=[numpy.get_include()],
        install_requires=REQUIREMENTS,
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3.11",
        ],
    )
