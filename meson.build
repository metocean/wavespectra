project('wavespectra', 'c',
    version : '3.9.0',
)

add_languages('fortran')

py = import('python').find_installation(pure: false)

py_dep = py.dependency()

incdir_numpy = run_command(py,
  ['-c', 'import os; import numpy; print(os.path.relpath(numpy.get_include()))'],
  check : true
).stdout().strip()

incdir_f2py = run_command(py,
    ['-c', 'import os; import numpy.f2py; print(os.path.relpath(numpy.f2py.get_include()))'],
    check : true
).stdout().strip()

specpart_source = custom_target('specpart.c',
  input : 'specpart.f90',  # .f so no F90 wrappers
  output : ['specpartmodule.c', 'specpart-f2pywrappers2.f90'],
  command : [py, '-m', 'numpy.f2py', '@INPUT@', '-m', 'specpart', '--lower']
)

inc_np = include_directories(incdir_numpy, incdir_f2py)
py.extension_module(
    'specpart',
    [
        'specpart.f90',
        specpart_source,
    ],
    incdir_f2py / 'fortranobject.c',
    include_directories: inc_np,
    dependencies : py_dep,
    install: true,
    subdir: 'specpart',
)

install_subdir('wavespectra', install_dir: py.get_install_dir())
