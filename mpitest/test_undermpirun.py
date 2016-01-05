
from __future__ import print_function

from unittest import TestCase

from openmdao.core.mpi_wrap import under_mpirun
from openmdao.test.mpi_util import MPITestCase


class SerialTestCase(TestCase):

    def test_not_under_mpirun(self):        
        self.assertFalse(under_mpirun())


class ParallelTestCase(MPITestCase):

    N_PROCS = 2

    def test_under_mpirun(self):        
        self.assertTrue(under_mpirun())




if __name__ == '__main__':
    from openmdao.test.mpi_util import mpirun_tests
    mpirun_tests()
