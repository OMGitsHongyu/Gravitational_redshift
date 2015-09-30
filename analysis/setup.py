from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extra_args = []
# Comment/Uncomment the following line to disable/enable OpenMP for GCC-ish
# compilers.
# extra_args = ["-fopenmp"]

exts = [Extension("calculator",
                  ["calculator.pyx"],
                  extra_compile_args=extra_args,
                  extra_link_args=extra_args),
        ]

setup(
    name = "Calculation",
    ext_modules = cythonize(exts),
)
