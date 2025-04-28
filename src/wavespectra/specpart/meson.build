py_mod = import('python').find_installation()
numpy_dep = dependency('numpy', required: true)

specpart_target = custom_target(
  'specpart',
  input: ['specpart.f90', 'specpart.pyf'],
  output: 'specpart.so',
  command: [
    py_mod, '-m', 'numpy.f2py',
    '-c', '@INPUT0@', '@INPUT1@',
    '-m', 'specpart',
  ],
  install: true,
  install_dir: py_mod.get_install_dir(),
)


