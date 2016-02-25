""" Unit test for the SqliteRecorder. """
import errno
import os

from shutil import rmtree
from tempfile import mkdtemp
import time

import numpy as np
from sqlitedict import SqliteDict

from openmdao.core.problem import Problem
from openmdao.core.group import Group
from openmdao.core.parallel_group import ParallelGroup
from openmdao.core.component import Component
from openmdao.core.mpi_wrap import MPI
from openmdao.components.indep_var_comp import IndepVarComp
from openmdao.recorders.sqlite_recorder import SqliteRecorder
from openmdao.recorders.test.test_sqlite import _assertMetadataRecorded, _assertIterationDataRecorded
from openmdao.test.mpi_util import MPITestCase

if MPI:
    from openmdao.core.petsc_impl import PetscImpl as impl
    coordinate = [MPI.COMM_WORLD.rank, 'Driver', (1, )]
else:
    from openmdao.core.basic_impl import BasicImpl as impl
    coordinate = [0, 'Driver', (1, )]


class ABCDArrayComp(Component):

    def __init__(self, arr_size=9, delay=0.01):
        super(ABCDArrayComp, self).__init__()
        self.add_param('a', np.ones(arr_size, float))
        self.add_param('b', np.ones(arr_size, float))
        self.add_param('in_string', '')
        self.add_param('in_list', [])

        self.add_output('c', np.ones(arr_size, float))
        self.add_output('d', np.ones(arr_size, float))
        self.add_output('out_string', '')
        self.add_output('out_list', [])

        self.delay = delay

    def solve_nonlinear(self, params, unknowns, resids):
        time.sleep(self.delay)

        unknowns['c'] = params['a'] + params['b']
        unknowns['d'] = params['a'] - params['b']

        unknowns['out_string'] = params['in_string'] + '_' + self.name
        unknowns['out_list']   = params['in_list'] + [1.5]


def run(problem):
    t0 = time.time()
    problem.run()
    t1 = time.time()

    return t0, t1


class TestSqliteRecorder(MPITestCase):
    filename = ""
    dir = ""
    N_PROCS = 2

    def setUp(self):
        self.dir = mkdtemp()
        self.filename = os.path.join(self.dir, "sqlite_test")
        self.tablename = 'openmdao'
        self.recorder = SqliteRecorder(self.filename)
        self.recorder.options['record_metadata'] = False
        self.eps = 1e-5

    def tearDown(self):
        try:
            rmtree(self.dir)
        except OSError as e:
            # If directory already deleted, keep going
            if e.errno not in (errno.ENOENT, errno.EACCES, errno.EPERM):
                raise e

    def assertMetadataRecorded(self, expected):
        if self.comm.rank != 0:
            return

        db = SqliteDict(self.filename, self.tablename)
        _assertMetadataRecorded(self, db, expected)
        db.close()

    def assertIterationDataRecorded(self, expected, tolerance, root):
        if self.comm.rank != 0:
            return

        db = SqliteDict(self.filename, self.tablename)
        _assertIterationDataRecorded(self, db, expected, tolerance)
        db.close()

    def test_basic(self):
        size = 3

        prob = Problem(Group(), impl=impl)

        G1 = prob.root.add('G1', ParallelGroup())
        G1.add('P1', IndepVarComp('x', np.ones(size, float) * 1.0))
        G1.add('P2', IndepVarComp('x', np.ones(size, float) * 2.0))

        prob.root.add('C1', ABCDArrayComp(size))

        prob.root.connect('G1.P1.x', 'C1.a')
        prob.root.connect('G1.P2.x', 'C1.b')

        prob.driver.add_recorder(self.recorder)

        self.recorder.options['record_params'] = True
        self.recorder.options['record_resids'] = True

        prob.setup(check=False)

        t0, t1 = run(prob)
        prob.cleanup()

        expected_params = [
            ("C1.a", [1.0, 1.0, 1.0]),
            ("C1.b", [2.0, 2.0, 2.0]),
        ]

        expected_unknowns = [
            ("G1.P1.x", np.array([1.0, 1.0, 1.0])),
            ("G1.P2.x", np.array([2.0, 2.0, 2.0])),
            ("C1.c",    np.array([3.0, 3.0, 3.0])),
            ("C1.d",    np.array([-1.0, -1.0, -1.0])),
            ("C1.out_string", "_C1"),
            ("C1.out_list", [1.5]),
        ]
        expected_resids = [
            ("G1.P1.x", np.array([0.0, 0.0, 0.0])),
            ("G1.P2.x", np.array([0.0, 0.0, 0.0])),
            ("C1.c",    np.array([0.0, 0.0, 0.0])),
            ("C1.d",    np.array([0.0, 0.0, 0.0])),
            ("C1.out_string", ""),
            ("C1.out_list", []),
        ]

        self.assertIterationDataRecorded(((coordinate, (t0, t1),
                                           expected_params, expected_unknowns,
                                           expected_resids),),
                                           self.eps, prob.root)

    def test_includes(self):
        size = 3

        prob = Problem(Group(), impl=impl)

        G1 = prob.root.add('G1', ParallelGroup())
        G1.add('P1', IndepVarComp('x', np.ones(size, float) * 1.0))
        G1.add('P2', IndepVarComp('x', np.ones(size, float) * 2.0))

        prob.root.add('C1', ABCDArrayComp(size))

        prob.root.connect('G1.P1.x', 'C1.a')
        prob.root.connect('G1.P2.x', 'C1.b')

        prob.driver.add_recorder(self.recorder)
        self.recorder.options['record_params'] = True
        self.recorder.options['record_resids'] = True

        self.recorder.options['includes'] = ['C1.*']
        prob.setup(check=False)

        t0, t1 = run(prob)
        prob.cleanup()

        expected_params = [
            ("C1.a", [1.0, 1.0, 1.0]),
            ("C1.b", [2.0, 2.0, 2.0]),
        ]
        expected_unknowns = [
            ("C1.c", np.array([3.0, 3.0, 3.0])),
            ("C1.d", np.array([-1.0, -1.0, -1.0])),
            ("C1.out_string", "_C1"),
            ("C1.out_list", [1.5]),
        ]
        expected_resids = [
            ("C1.c", np.array([0.0, 0.0, 0.0])),
            ("C1.d", np.array([0.0, 0.0, 0.0])),
            ("C1.out_string", ""),
            ("C1.out_list", []),
        ]

        self.assertIterationDataRecorded(((coordinate, (t0, t1), expected_params, expected_unknowns, expected_resids),), self.eps, prob.root)

    def test_includes_and_excludes(self):
        size = 3

        prob = Problem(Group(), impl=impl)

        G1 = prob.root.add('G1', ParallelGroup())
        G1.add('P1', IndepVarComp('x', np.ones(size, float) * 1.0))
        G1.add('P2', IndepVarComp('x', np.ones(size, float) * 2.0))

        prob.root.add('C1', ABCDArrayComp(size))

        prob.root.connect('G1.P1.x', 'C1.a')
        prob.root.connect('G1.P2.x', 'C1.b')

        prob.driver.add_recorder(self.recorder)

        self.recorder.options['includes'] = ['C1.*']
        self.recorder.options['excludes'] = ['*.out*']
        self.recorder.options['record_params'] = True
        self.recorder.options['record_resids'] = True

        prob.setup(check=False)

        t0, t1 = run(prob)
        prob.cleanup()

        expected_params = [
            ("C1.a", [1.0, 1.0, 1.0]),
            ("C1.b", [2.0, 2.0, 2.0]),
        ]

        expected_unknowns = [
            ("C1.c", np.array([3.0, 3.0, 3.0])),
            ("C1.d", np.array([-1.0, -1.0, -1.0])),
        ]

        expected_resids = [
            ("C1.c", np.array([0.0, 0.0, 0.0])),
            ("C1.d", np.array([0.0, 0.0, 0.0])),
        ]

        self.assertIterationDataRecorded(((coordinate, (t0, t1), expected_params, expected_unknowns, expected_resids),), self.eps, prob.root)

    def test_solver_record(self):
        size = 3

        prob = Problem(Group(), impl=impl)

        G1 = prob.root.add('G1', ParallelGroup())
        G1.add('P1', IndepVarComp('x', np.ones(size, float) * 1.0))
        G1.add('P2', IndepVarComp('x', np.ones(size, float) * 2.0))

        prob.root.add('C1', ABCDArrayComp(size))

        prob.root.connect('G1.P1.x', 'C1.a')
        prob.root.connect('G1.P2.x', 'C1.b')
        prob.root.nl_solver.add_recorder(self.recorder)
        self.recorder.options['record_params'] = True
        self.recorder.options['record_resids'] = True
        prob.setup(check=False)
        t0, t1 = run(prob)
        prob.cleanup()

        if MPI:
            coord = [MPI.COMM_WORLD.rank, 'Driver', (1, ), "root", (1,)]
        else:
            coord = [0, 'Driver', (1, ), "root", (1,)]

        expected_params = [
            ("C1.a", [1.0, 1.0, 1.0]),
            ("C1.b", [2.0, 2.0, 2.0]),
        ]
        expected_unknowns = [
            ("G1.P1.x", np.array([1.0, 1.0, 1.0])),
            ("G1.P2.x", np.array([2.0, 2.0, 2.0])),
            ("C1.c",    np.array([3.0, 3.0, 3.0])),
            ("C1.d",    np.array([-1.0, -1.0, -1.0])),
            ("C1.out_string", "_C1"),
            ("C1.out_list", [1.5]),
        ]
        expected_resids = [
            ("G1.P1.x", np.array([0.0, 0.0, 0.0])),
            ("G1.P2.x", np.array([0.0, 0.0, 0.0])),
            ("C1.c",    np.array([0.0, 0.0, 0.0])),
            ("C1.d",    np.array([0.0, 0.0, 0.0])),
            ("C1.out_string", ""),
            ("C1.out_list", []),
        ]

        self.assertIterationDataRecorded(((coord, (t0, t1), expected_params, expected_unknowns, expected_resids),), self.eps, prob.root)

    def test_driver_records_metadata(self):
        size = 3

        prob = Problem(Group(), impl=impl)

        G1 = prob.root.add('G1', ParallelGroup())
        G1.add('P1', IndepVarComp('x', np.ones(size, float) * 1.0))
        G1.add('P2', IndepVarComp('x', np.ones(size, float) * 2.0))

        prob.root.add('C1', ABCDArrayComp(size))

        prob.root.connect('G1.P1.x', 'C1.a')
        prob.root.connect('G1.P2.x', 'C1.b')

        prob.driver.add_recorder(self.recorder)

        self.recorder.options['record_metadata'] = True
        prob.setup(check=False)

        prob.cleanup()

        expected = (
            list(prob.root.params.iteritems()),
            list(prob.root.unknowns.iteritems()),
            list(prob.root.resids.iteritems()),
        )

        self.assertMetadataRecorded(expected)

    def test_driver_doesnt_records_metadata(self):
        size = 3

        prob = Problem(Group(), impl=impl)

        G1 = prob.root.add('G1', ParallelGroup())
        G1.add('P1', IndepVarComp('x', np.ones(size, float) * 1.0))
        G1.add('P2', IndepVarComp('x', np.ones(size, float) * 2.0))

        prob.root.add('C1', ABCDArrayComp(size))

        prob.root.connect('G1.P1.x', 'C1.a')
        prob.root.connect('G1.P2.x', 'C1.b')

        prob.driver.add_recorder(self.recorder)

        self.recorder.options['record_metadata'] = False
        prob.setup(check=False)

        prob.cleanup()

        self.assertMetadataRecorded(None)


if __name__ == "__main__":
    from openmdao.test.mpi_util import mpirun_tests
    mpirun_tests()
