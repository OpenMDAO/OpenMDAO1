import time
import os
from tempfile import mkdtemp
from shutil import rmtree

from six import text_type

import numpy as np

from openmdao.api import Problem, Group, ParallelGroup, Component, IndepVarComp, FileRef
from openmdao.core.mpi_wrap import MPI, FakeComm
from openmdao.test.mpi_util import MPITestCase

if MPI:
    from openmdao.core.petsc_impl import PetscImpl as impl
else:
    from openmdao.api import BasicImpl as impl

from openmdao.test.util import assert_rel_error


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

class PBOComp(Component):

    def __init__(self):
        super(PBOComp, self).__init__()
        self.add_param('a', [0.,0.,0.,0.,0.])
        self.add_param('b', [1.,2.,3.,4.,5.])

        self.add_output('c', [1.,2.,3.,4.,5.])
        self.add_output('d', [-1.,-2.,-3.,-4.,-5.])

    def solve_nonlinear(self, params, unknowns, resids):
        for i in range(5):
            unknowns['c'][i] = params['a'][i] + params['b'][i]
            unknowns['d'][i] = params['a'][i] - params['b'][i]

class FileSrc(Component):
    def __init__(self, name):
        super(FileSrc, self).__init__()
        self.add_output("fout", FileRef(name+'.out'))

    def solve_nonlinear(self, params, unknowns, resids):
        with self.unknowns['fout'].open('w') as f:
            f.write("%s\n" % self.pathname)

class FileMid(Component):
    def __init__(self, name):
        super(FileMid, self).__init__()
        self.add_param("fin", FileRef(name+'.in'))
        self.add_output("fout", FileRef(name+'.out'))

    def solve_nonlinear(self, params, unknowns, resids):
        with self.params['fin'].open('r') as fin, \
                      self.unknowns['fout'].open('w') as fout:
            fout.write(fin.read())
            fout.write("%s\n" % self.pathname)

class FileSink(Component):
    def __init__(self, name):
        super(FileSink, self).__init__()
        self.add_param("fin1", FileRef(name+'1.in'))
        self.add_param("fin2", FileRef(name+'2.in'))

    def solve_nonlinear(self, params, unknowns, resids):
        pass

class PBOTestCase(MPITestCase):
    N_PROCS=1

    def test_simple(self):
        prob = Problem(Group(), impl=impl)

        A1 = prob.root.add('A1', IndepVarComp('a', [1.,1.,1.,1.,1.]))
        B1 = prob.root.add('B1', IndepVarComp('b', [1.,1.,1.,1.,1.]))

        C1 = prob.root.add('C1', PBOComp())
        C2 = prob.root.add('C2', PBOComp())

        prob.root.connect('A1.a', 'C1.a')
        prob.root.connect('B1.b', 'C1.b')

        prob.root.connect('C1.c', 'C2.a')
        prob.root.connect('C1.d', 'C2.b')

        prob.setup(check=False)

        prob.run()

        self.assertEqual(prob['C2.a'], [2.,2.,2.,2.,2.])
        self.assertEqual(prob['C2.b'], [0.,0.,0.,0.,0.])
        self.assertEqual(prob['C2.c'], [2.,2.,2.,2.,2.])
        self.assertEqual(prob['C2.d'], [2.,2.,2.,2.,2.])
        self.assertEqual(prob.root.unknowns.vec.size, 0)

class PBOTestCase2(MPITestCase):
    N_PROCS=2

    def test_fan_in(self):
        prob = Problem(Group(), impl=impl)
        par = prob.root.add('par', ParallelGroup())

        G1 = par.add('G1', Group())
        A1 = G1.add('A1', IndepVarComp('a', [1.,1.,1.,1.,1.]))
        C1 = G1.add('C1', PBOComp())

        G2 = par.add('G2', Group())
        B1 = G2.add('B1', IndepVarComp('b', [3.,3.,3.,3.,3.]))
        C2 = G2.add('C2', PBOComp())

        C3 = prob.root.add('C3', PBOComp())

        par.connect('G1.A1.a', 'G1.C1.a')
        par.connect('G2.B1.b', 'G2.C2.a')
        prob.root.connect('par.G1.C1.c', 'C3.a')
        prob.root.connect('par.G2.C2.c', 'C3.b')

        prob.setup(check=False)

        prob.run()

        self.assertEqual(prob['C3.a'], [2.,3.,4.,5.,6.])
        self.assertEqual(prob['C3.b'], [4.,5.,6.,7.,8.])
        self.assertEqual(prob['C3.c'], [6.,8.,10.,12.,14.])
        self.assertEqual(prob['C3.d'], [-2.,-2.,-2.,-2.,-2.])
        self.assertEqual(prob.root.unknowns.vec.size, 0)


class FileRefTestCase(MPITestCase):
    N_PROCS=2

    def setUp(self):
        self.startdir = os.getcwd()
        self.tmpdir = mkdtemp()
        os.chdir(self.tmpdir)

    def tearDown(self):
        os.chdir(self.startdir)
        try:
            rmtree(self.tmpdir)
        except OSError as e:
            # If directory already deleted, keep going
            if e.errno != errno.ENOENT:
                raise e

    def test_file_diamond(self):
        prob = Problem(Group(), impl=impl)
        src = prob.root.add("src", FileSrc('src'))
        par = prob.root.add('par', ParallelGroup())
        par.add("mid1", FileMid('mid1'))
        par.add("mid2", FileMid('mid2'))
        sink = prob.root.add("sink", FileSink('sink'))

        prob.root.connect('src.fout', 'par.mid1.fin')
        prob.root.connect('src.fout', 'par.mid2.fin')
        prob.root.connect('par.mid1.fout', 'sink.fin1')
        prob.root.connect('par.mid2.fout', 'sink.fin2')

        prob.setup(check=False)
        prob.run()

        with sink.params['fin1'].open('r') as f:
            self.assertEqual(f.read(), "src\npar.mid1\n")
        with sink.params['fin2'].open('r') as f:
            self.assertEqual(f.read(), "src\npar.mid2\n")


class MPITests1(MPITestCase):

    N_PROCS = 1

    def test_comm(self):
        prob = Problem(Group(), impl=impl)
        prob.setup(check=False)

        if MPI:
            assert prob.comm is MPI.COMM_WORLD
        else:
            assert isinstance(prob.comm, FakeComm)

    def test_simple(self):
        prob = Problem(Group(), impl=impl)

        size = 5
        A1 = prob.root.add('A1', IndepVarComp('a', np.zeros(size, float)))
        B1 = prob.root.add('B1', IndepVarComp('b', np.zeros(size, float)))
        B2 = prob.root.add('B2', IndepVarComp('b', np.zeros(size, float)))
        S1 = prob.root.add('S1', IndepVarComp('s', ''))
        L1 = prob.root.add('L1', IndepVarComp('l', []))

        C1 = prob.root.add('C1', ABCDArrayComp(size))
        C2 = prob.root.add('C2', ABCDArrayComp(size))

        prob.root.connect('A1.a', 'C1.a')
        prob.root.connect('B1.b', 'C1.b')
        prob.root.connect('S1.s', 'C1.in_string')
        prob.root.connect('L1.l', 'C1.in_list')

        prob.root.connect('C1.c', 'C2.a')
        prob.root.connect('B2.b', 'C2.b')
        prob.root.connect('C1.out_string', 'C2.in_string')
        prob.root.connect('C1.out_list',   'C2.in_list')

        prob.setup(check=False)

        prob['A1.a'] = np.ones(size, float) * 3.0
        prob['B1.b'] = np.ones(size, float) * 7.0
        prob['B2.b'] = np.ones(size, float) * 5.0

        prob.run()

        self.assertTrue(all(prob['C2.a'] == np.ones(size, float)*10.))
        self.assertTrue(all(prob['C2.b'] == np.ones(size, float)*5.))
        self.assertTrue(all(prob['C2.c'] == np.ones(size, float)*15.))
        self.assertTrue(all(prob['C2.d'] == np.ones(size, float)*5.))

        self.assertTrue(prob['C2.out_string']=='_C1_C2')
        self.assertTrue(prob['C2.out_list']==[1.5, 1.5])


class MPITests2(MPITestCase):

    N_PROCS = 2

    def test_too_many_procs(self):
        prob = Problem(Group(), impl=impl)

        size = 5
        A1 = prob.root.add('A1', IndepVarComp('a', np.zeros(size, float)))
        C1 = prob.root.add('C1', ABCDArrayComp(size))

        try:
            prob.setup(check=False)
        except Exception as err:
            self.assertEqual(str(err),
                             "This problem was given 2 MPI processes, "
                             "but it requires between 1 and 1.")
        else:
            if MPI:
                self.fail("Exception expected")

    def test_parallel_fan_in(self):
        size = 3

        prob = Problem(Group(), impl=impl)

        G1 = prob.root.add('G1', ParallelGroup())
        G1.add('P1', IndepVarComp('x', np.ones(size, float) * 1.0))
        G1.add('P2', IndepVarComp('x', np.ones(size, float) * 2.0))

        prob.root.add('C1', ABCDArrayComp(size))

        prob.root.connect('G1.P1.x', 'C1.a')
        prob.root.connect('G1.P2.x', 'C1.b')

        prob.setup(check=False)
        prob.run()

        if not MPI or self.comm.rank == 0:
            self.assertTrue(all(prob.root.C1.params['a'] == np.ones(size, float)*1.0))
            self.assertTrue(all(prob.root.C1.params['b'] == np.ones(size, float)*2.0))
            self.assertTrue(all(prob['C1.c'] == np.ones(size, float)*3.0))
            self.assertTrue(all(prob['C1.d'] == np.ones(size, float)*-1.0))
            # TODO: not handling non-flattenable vars yet

        if MPI and self.comm.rank == 1:
            # check for useful error messages when trying to get/set remote variable
            try:
                x = prob['G1.P1.x']
            except Exception as error:
                msg = "Cannot access remote Variable 'G1.P1.x' in this process."
                self.assertEqual(text_type(error), msg)
            else:
                self.fail("Error expected")

            try:
                prob['G1.P1.x'] = 0.
            except Exception as error:
                msg = "Cannot access remote Variable 'G1.P1.x' in this process."
                self.assertEqual(text_type(error), msg)
            else:
                self.fail("Error expected")

    def test_parallel_diamond(self):
        size = 3
        prob = Problem(Group(), impl=impl)
        root = prob.root
        root.add('P1', IndepVarComp('x', np.ones(size, float) * 1.1))
        G1 = root.add('G1', ParallelGroup())
        G1.add('C1', ABCDArrayComp(size))
        G1.add('C2', ABCDArrayComp(size))
        root.add('C3', ABCDArrayComp(size))

        root.connect('P1.x', 'G1.C1.a')
        root.connect('P1.x', 'G1.C2.b')
        root.connect('G1.C1.c', 'C3.a')
        root.connect('G1.C2.d', 'C3.b')

        prob.setup(check=False)
        prob.run()

        if not MPI or self.comm.rank == 0:
            assert_rel_error(self, prob.root.G1.C1.unknowns['c'],
                             np.ones(size)*2.1, 1.e-10)
            assert_rel_error(self, prob.root.G1.C1.unknowns['d'],
                             np.ones(size)*.1, 1.e-10)
            assert_rel_error(self, prob.root.C3.params['a'],
                             np.ones(size)*2.1, 1.e-10)
            assert_rel_error(self, prob.root.C3.params['b'],
                             np.ones(size)*-.1, 1.e-10)

        if not MPI or self.comm.rank == 1:
            assert_rel_error(self, prob.root.G1.C2.unknowns['c'],
                             np.ones(size)*2.1, 1.e-10)
            assert_rel_error(self, prob.root.G1.C2.unknowns['d'],
                             np.ones(size)*-.1, 1.e-10)

    def test_wrong_impl(self):
        if MPI:
            try:
                Problem(Group())
            except Exception as err:
                self.assertEqual(str(err), "To run under MPI, the impl for"
                                           " a Problem must be PetscImpl.")
            else:
                self.fail("Exception expected")

    def test_multiple_problems(self):
        if MPI:
            # split the comm and run an instance of the Problem in each subcomm
            subcomm = self.comm.Split(self.comm.rank)
            prob = Problem(Group(), impl=impl, comm=subcomm)

            size = 5
            value = self.comm.rank + 1
            values = np.ones(size)*value

            A1 = prob.root.add('A1', IndepVarComp('x', values))
            C1 = prob.root.add('C1', ABCDArrayComp(size))

            prob.root.connect('A1.x', 'C1.a')
            prob.root.connect('A1.x', 'C1.b')

            prob.setup(check=False)
            prob.run()

            # check the first output array and store in result
            self.assertTrue(all(prob['C1.c'] == np.ones(size)*(value*2)))
            result = prob['C1.c']

            # gather the results from the separate processes/problems and check
            # for expected values
            results = self.comm.allgather(result)
            self.assertEqual(len(results), self.comm.size)

            for n in range(self.comm.size):
                expected = np.ones(size)*2*(n+1)
                self.assertTrue(all(results[n] == expected))

if __name__ == '__main__':
    from openmdao.test.mpi_util import mpirun_tests
    mpirun_tests()
