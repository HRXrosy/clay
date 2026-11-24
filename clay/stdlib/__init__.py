"""
Clay standard library.

Importing any submodule auto-registers its package in the Clay module
registry. Users prefer the higher-level lookup:

    from clay.module import resolve
    opamps = resolve("clay.stdlib.opamps")
    cell = opamps.TwoStageMillerPMOS(spec=my_spec)
"""
