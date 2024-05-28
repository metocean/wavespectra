project('wavespectra', 'fortran', version : '3.9.0-dev')

py = import('python').find_installation(pure: false)

py.extension_module(
    'specpart',
    'wavespectra/specpart/specpart.f90',
    install: true,
)

