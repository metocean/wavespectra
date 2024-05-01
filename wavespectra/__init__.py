"""Define module attributes.

- Defining packaging attributes accessed by setup.py
- Making reading functions available at module level

"""

__version__ = "3.9.0"
__author__ = "MetOcean Solutions"
__contact__ = "support@metocean.co.nz"
__url__ = "http://github.com/metocean/wavespectra"
__description__ = "Ocean wave spectra tools"
__keywords__ = "wave spectra ocean xarray statistics analysis"


def _import_read_functions(pkgname="input"):
    """Make read functions available at module level.

    Functions are imported here if:
        - they are defined in a module wavespectra.input.{modname}
        - they are named as read_{modname}

    """
    import glob
    import os
    import sys
    from importlib import import_module

    here = os.path.dirname(os.path.abspath(__file__))
    for filename in glob.glob1(os.path.join(here, pkgname), "*.py"):
        module = os.path.splitext(filename)[0]
        if module == "__init__":
            continue
        func_name = "read_{}".format(module)
        try:
            globals()[func_name] = getattr(
                import_module("wavespectra.{}.{}".format(pkgname, module)), func_name
            )
        except Exception as exc:
            print("Cannot import reading function {}:\n{}".format(func_name, exc))


_import_read_functions()
