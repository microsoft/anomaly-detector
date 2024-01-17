from Cython.Build import cythonize

def build(setup_kwargs):
    
    compiler_directives = {"language_level": 3, "embedsignature": True}
    setup_kwargs.update(
        {
            "name": "anomaly_detector",
            "package": ["anomaly_detector"],
            # https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#cythonize-arguments
            "ext_modules": cythonize(
                module_list="anomaly_detector/univariate/*.pyx",
                compiler_directives=compiler_directives,
                nthreads=5,
            ),
        }
    )