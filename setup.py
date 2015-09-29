from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extra_args = []
# Comment/Uncomment the following line to disable/enable OpenMP for GCC-ish
# compilers.
# extra_args = ["-fopenmp"]

exts = [Extension("pot_bf",
                  ["pot_bf.pyx"],
                  extra_compile_args=extra_args,
                  extra_link_args=extra_args),
        ]

setup(
    name = "Potential calculation by brute force",
    ext_modules = cythonize(exts),
)
