from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

import os
import sys

# Include the pybind11 header files
import pybind11


os.environ["CC"] = "/etc/alternatives/gcc"
os.environ["CXX"] = "/etc/alternatives/g++"

ext_modules = [
    Extension(
        "fast_dbscan",
        ["bindings.cpp", "dbscan.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        #extra_compile_args=["-std=c++20", "-Ivendor/"],
        # extra_link_args=["-lstdc++"],
	extra_compile_args=["-std=c++20", "-Ivendor/"],
	# extra_link_args=["-stdlib=libc++", "-lstdc++"],
	# cxx_std=20
    )
]

class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        "msvc": ["/EHsc"],
        "unix": [],
    }

    if sys.platform == "darwin":
        c_opts["unix"] += ["-stdlib=libc++", "-mmacosx-version-min=10.7"]

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        for ext in self.extensions:
            ext.extra_compile_args = opts
            print(ext)
        build_ext.build_extensions(self)

setup(
    name="fast_dbscan",
    ext_modules=ext_modules,
#     cmdclass={"build_ext": BuildExt},
    zip_safe=False,
)
