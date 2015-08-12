""" Testing out MPI optimization with pyopt_sparse"""

import unittest
import numpy as np

from openmdao.core.mpi_wrap import MPI

if MPI:
    from openmdao.core.petsc_impl import PetscImpl as impl
else:
    from openmdao.core.basic_impl import BasicImpl as impl

SKIP = False
try:
    from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver
except ImportError:
    # Just so python can parse this file.
    from openmdao.core.driver import Driver
    pyOptSparseDriver = Driver
    SKIP = True


if __name__ == '__main__':
    from openmdao.test.mpi_util import mpirun_tests
    mpirun_tests()
