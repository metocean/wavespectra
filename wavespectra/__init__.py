"""Define module attributes.

- Defining packaging attributes accessed by setup.py
- Making reading functions available at module level

"""

__version__ = "3.8.1"
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
    from importlib import import_module

    here = os.path.dirname(os.path.abspath(__file__))
    for filename in glob.glob1(os.path.join(here, pkgname), "*.py"):
        module = os.path.splitext(filename)[0]
        if module == "__init__":
            continue
        func_name = f"read_{module}"
        try:
            globals()[func_name] = getattr(
                import_module(f"wavespectra.{pkgname}.{module}"), func_name
            )
        except Exception as exc:
            print(f"Cannot import reading function {func_name}:\n{exc}")


_import_read_functions()
