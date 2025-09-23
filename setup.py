from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import subprocess
import os
import pathlib
import sys

def readme():
    with open('README.md') as f:
        return f.read()

def rootcmd(flag):
    return [subprocess.check_output(["root-config", flag]).decode("utf-8").rstrip("\n")]

def rootinc():
    return rootcmd("--incdir")

def rootlib():
    return rootcmd("--libdir")

def numpyinc():
    import numpy
    return [numpy.get_include()]

def usrinc(): 
    dirs = [
	"/usr/include/"
        "/usr/local/include/",
        "/opt/homebrew/include/"
    ]
    if "BOOST_INC" in os.environ:
        dirs.append(os.environ["BOOST_INC"])
	
    return [d for d in dirs if os.path.exists(d)]

def localinc():
    dirs = [
	'{CMAKE_INSTALL}include',
	'{CMAKE_INSTALL}include/eigen3',
    ]
    return dirs 

def locallib():
    dirs = [
        "{CMAKE_INSTALL}lib",
	"{CMAKE_INSTALL}lib64",
	"/opt/homebrew/lib"
    ]
    return dirs

def scripts():
    thisdir = pathlib.Path(__file__).parent.resolve()
    return ["profit/bin/" + f for f in os.listdir(thisdir/"profit"/"bin") if str(f).endswith(".py") or str(f).startswith("PRO")]

# Set number of workers to half number of available cores
def njob():
    cores = os.cpu_count()
    return cores // 2

# From: https://stackoverflow.com/questions/42585210/extending-setuptools-extension-to-use-cmake-in-setup-py
class CMakeExtension(Extension):
    def __init__(self, name):
        # don't invoke the original build_ext for this special extension
        super().__init__(name, sources=[])

class build_ext_wcmake(build_ext):
    def run(self):
        self.cmake_install = ""

        for ext in self.extensions:
            if isinstance(ext, CMakeExtension):
                self.build_cmake(ext)

        # Set CMake install location in extensions
        for ext in self.extensions:
            ext.include_dirs = [s.replace("{CMAKE_INSTALL}", self.cmake_install) for s in ext.include_dirs]
            ext.library_dirs = [s.replace("{CMAKE_INSTALL}", self.cmake_install) for s in ext.library_dirs]

        super().run()

    def build_cmake(self, ext):
        cwd = pathlib.Path().absolute()

        # these dirs will be created in build_py, so if you don't have
        # any python sources to bundle, the dirs will be missing
        build_temp = pathlib.Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
        extdir = pathlib.Path(self.get_ext_fullpath(ext.name))
        extdir.mkdir(parents=True, exist_ok=True)

        # example of cmake args
        config = 'Debug' if self.debug else 'Release'
        cmake_args = [
            '-DCMAKE_INSTALL_PREFIX=' + str(extdir.parent.absolute()),
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + str(extdir.parent.absolute()),
            '-DCMAKE_BUILD_TYPE=' + config
        ]

        # Tell pybind where we are installing everything
        self.cmake_install = str(extdir.parent.absolute()) + "/"

        # example of build args
        build_args = [
            '--config', config,
            '-j', '%i' % njob()
        ]

        os.chdir(str(build_temp))
        self.spawn(['cmake', str(cwd)] + cmake_args)
        if not self.dry_run:
            self.spawn(['cmake', '--build', '.'] + build_args)
            self.spawn(['cmake', '--install', '.'] + build_args[:-2])

        # Troubleshooting: if fail on line above then delete all possible 
        # temporary CMake files including "CMakeCache.txt" in top level dir.
        os.chdir(str(cwd))

ext_modules = [
    CMakeExtension('profit/PROfit'),
    Pybind11Extension(
        '_profit',
        ['pybind/profit.cxx'],
        include_dirs=usrinc() + localinc() + rootinc() + numpyinc(),
        library_dirs=locallib() + rootlib(),
        libraries=["Core", "PROfitLib", "tinyxml2", "TreePlayer", "boost_serialization"],
        language='c++',
        cxx_std=17
    )
]

# Dependencies for different python versions
deps_py3_9 =[
        'pandas==2.2.3',
        'uproot==5.5.1',
        'awkward==2.7.2',
	'tables==3.9.2'
]
    
deps_py3_10=[
        'pandas==2.2.3',
        'uproot==5.5.1',
        'awkward==2.7.2',
	'tables==3.10.1'
]

deps = deps_py3_10 if sys.version_info[1] >= 10 else deps_py3_9

setup(
    name='profit',
    ext_modules=ext_modules,
    version='0.1',
    description='Python library for PROfit fitting package.',
    url='https://github.com/gputnam/Elephant_Vanishes/',
    author='Gray Putnam',
    author_email='gputnam@fnal.gov',
    license='MIT',
    packages=['profit'],
    scripts=scripts(),
    cmdclass={'build_ext': build_ext_wcmake},
    python_requires='>=3.9.15',
    install_requires=deps,
    zip_safe=False)
