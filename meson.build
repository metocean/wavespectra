project( 'wavespectra', 'c',
  # Note that the git commit hash cannot be added dynamically here
  version: '3.8.2',
  license: 'Copyright (c) 2018 MetOcean Solutions Ltd',
  meson_version: '>= 0.64.0',
  default_options: [
    'buildtype=debugoptimized',
    # TODO: the below -Wno flags are all needed to silence warnings in
    # f2py-generated code. This should be fixed in f2py itself.
    'c_args=-Wno-unused-function -Wno-conversion -Wno-misleading-indentation -Wno-incompatible-pointer-types',
    'fortran_args=-Wno-conversion',  # silence "conversion from REAL(8) to INTEGER(4)"
    'fortran_std=legacy',
  ],
)

# Adding at project level causes many spurious -lgfortran flags.
add_languages('fortran', native: false)


fc = meson.get_compiler('fortran')
cc = meson.get_compiler('c')

# https://mesonbuild.com/Python-module.html
py_mod = import('python')
py3 = py_mod.find_installation(pure: false)
py3_dep = py3.dependency()
message(py3.full_path())
message(py3.get_install_dir())

# Determine whether this is being compiled with numpy 1.X or 2.X
numpy_ver = run_command(py3,
  ['-c', 'import numpy; print(numpy.__version__.split(".")[0])'],
  check : true
).stdout().strip()
is_numpytwo = numpy_ver.to_int() >= 2

message('The Numpy major version is: ', numpy_ver)


# Don't use the deprecated NumPy C API. Define this to a fixed version instead
# of NPY_API_VERSION in order not to break compilation for released versions
# when NumPy introduces a new deprecation. Use in a meson.build file::
#
#   py3.extension_module('_name',
#     'source_fname',
#     numpy_nodepr_api)
#
if is_numpytwo
  numpy_nodepr_api = '-DNPY_NO_DEPRECATED_API=NPY_2_0_API_VERSION'
else
  numpy_nodepr_api = '-DNPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION'
endif

incdir_numpy = run_command(py3,
  ['-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())'],
  check : true
).stdout().strip()

incdir_f2py = run_command(py3,
  ['-c', 'import os; os.chdir(".."); from numpy import f2py; print(f2py.get_include())'],
  check : true
).stdout().strip()

specpart_source = custom_target('specpartmodule.c',
  input : ['src/specpart.f90'],  # .f so no F90 wrappers
  output : ['specpartmodule.c', 'specpart-f2pywrappers2.f90'],
  command : [py3, '-m', 'numpy.f2py', '@INPUT@', '-m', 'specpart', '--lower']
)

message('Include dirs are: ', incdir_f2py, ' ', incdir_numpy)
inc_dirs = include_directories(incdir_numpy, incdir_f2py)


# Platform detection to set more flags for Windows systems
is_windows = host_machine.system() == 'windows'
is_mac = host_machine.system() == 'darwin'
is_mingw = is_windows and cc.get_id() == 'gcc'

py3.extension_module('specpart',
  ['src/specpart.f90', specpart_source],
  incdir_f2py / 'fortranobject.c',
  include_directories: inc_dirs,
  dependencies: py3_dep,
  install: true,
  install_dir: join_paths(py3.get_install_dir(), 'wavespectra/specpart')
)

subdir('wavespectra')


