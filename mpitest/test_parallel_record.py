""" Testing out recorders under MPI."""

import os
import unittest

from openmdao.core import Problem
from openmdao.core.mpi_wrap import MPI
from openmdao.recorders import DumpRecorder
from openmdao.test.simple_comps import FanInGrouped
from openmdao.test.mpi_util import MPITestCase
from six import iteritems

if MPI: # pragma: no cover
    from openmdao.core.petsc_impl import PetscImpl as impl
else:
    from openmdao.core import BasicImpl as impl


class TestDumpRecorder(MPITestCase):

    N_PROCS = 2

    def setUp(self):
        if MPI: # pragma: no cover
            if MPI.COMM_WORLD.rank == 0:
                self.filename = 'data_0.dmp'
            elif MPI.COMM_WORLD.rank == 1:
                self.filename = 'data_1.dmp'
        else:
            self.filename = 'data.dmp'

    def tearDown(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def test_metadata_recorded(self):
        prob = Problem(impl=impl)
        prob.root = FanInGrouped()

        rec = DumpRecorder(out='data.dmp')
        rec.options['record_metadata'] = True
        rec.options['includes'] = ['p1.x1', 'p2.x2', 'comp3.y']
        prob.driver.add_recorder(rec)

        prob.setup(check=False)
        rec.close()
        
        with open(self.filename, 'r') as dumpfile:
            params = iteritems(prob.root.params)
            unknowns = iteritems(prob.root.unknowns)
            resids = iteritems(prob.root.resids)

            self.assertEqual("Metadata:\n", dumpfile.readline())
            self.assertEqual("Params:\n", dumpfile.readline())
            
            for name, metadata in params:
                fmat = "  {0}: {1}\n".format(name, metadata)
                self.assertEqual(fmat, dumpfile.readline())

            self.assertEqual("Unknowns:\n", dumpfile.readline())
            
            for name, metadata in unknowns:
                fmat = "  {0}: {1}\n".format(name, metadata)
                self.assertEqual(fmat, dumpfile.readline())
            
            self.assertEqual("Resids:\n", dumpfile.readline())

            for name, metadata in resids:
                fmat = "  {0}: {1}\n".format(name, metadata)
                self.assertEqual(fmat, dumpfile.readline())

    def test_dump_converge_diverge_par(self):

        prob = Problem(impl=impl)
        prob.root = FanInGrouped()

        rec = DumpRecorder(out='data.dmp')
        rec.options['record_params'] = True
        rec.options['record_resids'] = True
        rec.options['includes'] = ['p1.x1', 'p2.x2', 'comp3.y']
        prob.driver.add_recorder(rec)

        prob.setup(check=False)
        prob.run()

        with open(self.filename, 'r') as dumpfile:
            dump = dumpfile.readlines()

        self.assertEqual(dump[4], '  comp3.y: 29.0\n')
        self.assertEqual(dump[5], '  p1.x1: 1.0\n')
        self.assertEqual(dump[6], '  p2.x2: 1.0\n')

if __name__ == '__main__':
    from openmdao.test.mpi_util import mpirun_tests
    mpirun_tests()
