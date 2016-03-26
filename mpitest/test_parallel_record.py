""" Testing out recorders under MPI."""
import errno
import os

from shutil import rmtree
from tempfile import mkdtemp

from openmdao.core.problem import Problem
from openmdao.core.mpi_wrap import MPI
from openmdao.recorders.dump_recorder import DumpRecorder
from openmdao.test.simple_comps import FanInGrouped
from openmdao.test.mpi_util import MPITestCase
from six import iteritems

if MPI:
    from openmdao.core.petsc_impl import PetscImpl as impl
else:
    from openmdao.api import BasicImpl as impl


class TestDumpRecorder(MPITestCase):

    N_PROCS = 2

    def setUp(self):
        self.dir = mkdtemp()
        self.filename = os.path.join(self.dir, "data.dmp")

        if MPI:
            if MPI.COMM_WORLD.rank == 0:
                self.expected_filename = os.path.join(self.dir, 'data_0.dmp')
            elif MPI.COMM_WORLD.rank == 1:
                self.expected_filename = os.path.join(self.dir, 'data_1.dmp')
        else:
            self.expected_filename = os.path.join(self.dir, 'data.dmp')

    def tearDown(self):
        try:
            rmtree(self.dir)
        except OSError as e:
            # If directory already deleted, keep going
            if e.errno not in (errno.ENOENT, errno.EACCES, errno.EPERM):
                raise e

    def test_metadata_recorded(self):
        prob = Problem(impl=impl)
        prob.root = FanInGrouped()

        rec = DumpRecorder(out=self.filename)
        rec.options['record_metadata'] = True
        rec.options['includes'] = ['p1.x1', 'p2.x2', 'comp3.y']
        prob.driver.add_recorder(rec)

        prob.setup(check=False)
        prob.cleanup()

        with open(self.expected_filename, 'r') as dumpfile:
            params = iteritems(prob.root.params)
            unknowns = iteritems(prob.root.unknowns)

            self.assertEqual("Metadata:\n", dumpfile.readline())
            self.assertEqual("Params:\n", dumpfile.readline())

            for name, metadata in params:
                fmat = "  {0}: {1}\n".format(name, metadata)
                self.assertEqual(fmat, dumpfile.readline())

            self.assertEqual("Unknowns:\n", dumpfile.readline())

            for name, metadata in unknowns:
                fmat = "  {0}: {1}\n".format(name, metadata)
                self.assertEqual(fmat, dumpfile.readline())

    def test_dump_converge_diverge_par(self):

        prob = Problem(impl=impl)
        prob.root = FanInGrouped()

        rec = DumpRecorder(out=self.filename)
        rec.options['record_metadata'] = False
        rec.options['record_params'] = True
        rec.options['record_resids'] = True
        rec.options['includes'] = ['p1.x1', 'p2.x2', 'comp3.y']
        prob.driver.add_recorder(rec)

        prob.setup(check=False)
        prob.run()

        with open(self.expected_filename, 'r') as dumpfile:
            dump = dumpfile.readlines()

        if MPI and self.comm.rank == 0:  # rank 1 doesn't 'own' any vars
            self.assertEqual(dump[5], '  comp3.y: 29.0\n')
            self.assertEqual(dump[6], '  p1.x1: 1.0\n')
            self.assertEqual(dump[7], '  p2.x2: 1.0\n')


if __name__ == '__main__':
    from openmdao.test.mpi_util import mpirun_tests
    mpirun_tests()
