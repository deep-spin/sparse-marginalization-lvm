import os
import shutil

from setuptools import setup, find_packages
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext
from distutils.command.clean import clean

from Cython.Build import cythonize

from lpsmap import config


AD3_FLAGS_UNIX = [
    '-std=c++11',
    '-O3',
    '-Wall',
    '-Wno-sign-compare',
    '-Wno-overloaded-virtual',
    '-c',
    '-fmessage-length=0',
    '-fPIC',
    '-ffast-math',
    '-march=native'
]


AD3_FLAGS_MSVC = [
    '/O2',
    '/fp:fast',
    '/favor:INTEL64',
    '/wd4267'  # suppress sign-compare--like warning
]


AD3_CFLAGS = {
    'cygwin': AD3_FLAGS_UNIX,
    'mingw32': AD3_FLAGS_UNIX,
    'unix': AD3_FLAGS_UNIX,
    'msvc': AD3_FLAGS_MSVC
}


# support compiler-specific cflags in extensions and libs
class our_build_ext(build_ext):
    def build_extensions(self):

        # bug in distutils: flag not valid for c++
        flag = '-Wstrict-prototypes'
        if (hasattr(self.compiler, 'compiler_so')
                and flag in self.compiler.compiler_so):
            self.compiler.compiler_so.remove(flag)

        compiler_type = self.compiler.compiler_type
        compile_args = AD3_CFLAGS.get(compiler_type, [])

        for e in self.extensions:
            e.extra_compile_args.extend(compile_args)

        build_ext.build_extensions(self)


class our_clean(clean):
    def run(self):

        if os.path.exists('build'):
            shutil.rmtree('build')

        for dirpath, dirnames, filenames in os.walk('.'):
            for filename in filenames:
                if any(filename.endswith(suffix) for suffix in
                       (".so", ".pyd", ".dll", ".pyc")):
                    os.unlink(os.path.join(dirpath, filename))
                    continue
                extension = os.path.splitext(filename)[1]
                if extension in ['.c', '.cpp']:
                    pyx_file = str.replace(filename, extension, '.pyx')
                    if os.path.exists(os.path.join(dirpath, pyx_file)):
                        os.unlink(os.path.join(dirpath, filename))
            for dirname in dirnames:
                if dirname == '__pycache__':
                    shutil.rmtree(os.path.join(dirpath, dirname))
        clean.run(self)


# this is a backport of a workaround for a problem in distutils.

cmdclass = {'build_ext': our_build_ext,
            'clean': our_clean}


extensions = [
    Extension('lvmhelpers.pbinary_topk',
              ["lvmhelpers/pbinary_topk.pyx"]),
    Extension('lvmhelpers.pbernoulli',
              ["lvmhelpers/pbernoulli.pyx"],
              libraries=['ad3'],
              library_dirs=[config.get_libdir()],
              include_dirs=[config.get_include()]),
    Extension('lvmhelpers.pbudget',
              ["lvmhelpers/pbudget.pyx"],
              libraries=['ad3'],
              library_dirs=[config.get_libdir()],
              include_dirs=[config.get_include()]),
    Extension('lvmhelpers.psequence',
              ["lvmhelpers/psequence.pyx"],
              libraries=['ad3'],
              library_dirs=[config.get_libdir()],
              include_dirs=[config.get_include()]),
]

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='lvmhelpers',
    version='0.3',
    url='https://github.com/goncalomcorreia/explicit-sparse-marginalization',
    author='Gonçalo Correia',
    author_email='goncalommac@gmail.com',
    packages=find_packages(),
    install_requires=requirements,
    cmdclass=cmdclass,
    include_package_data=True,
    ext_modules=cythonize(extensions)
)
