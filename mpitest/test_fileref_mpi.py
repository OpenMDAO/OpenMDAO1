import os
from tempfile import mkdtemp
from shutil import rmtree

from openmdao.api import Problem, Group, ParallelGroup, Component, FileRef
from openmdao.core.mpi_wrap import MPI
from openmdao.test.mpi_util import MPITestCase

if MPI:
    from openmdao.core.petsc_impl import PetscImpl as impl
else:
    from openmdao.api import BasicImpl as impl


class FileSrc(Component):
    def __init__(self, name):
        super(FileSrc, self).__init__()
        self.add_output("fout", FileRef(name+'.dat'))

    def solve_nonlinear(self, params, unknowns, resids):
        with self.unknowns['fout'].open('w') as f:
            f.write("%s\n" % self.pathname)

class FileMid(Component):
    def __init__(self, iname, oname):
        super(FileMid, self).__init__()
        self.add_param("fin", FileRef(iname+'.dat'))
        self.add_output("fout", FileRef(oname+'.out'))

    def solve_nonlinear(self, params, unknowns, resids):
        with self.params['fin'].open('r') as fin, \
                      self.unknowns['fout'].open('w') as fout:
            fout.write(fin.read())
            fout.write("%s\n" % self.pathname)

class FileSink(Component):
    def __init__(self, name, num_ins):
        super(FileSink, self).__init__()
        for i in range(num_ins):
            self.add_param("fin%d"%i, FileRef(name+'%d.in'%i))

    def solve_nonlinear(self, params, unknowns, resids):
        pass

class FileRefTestCase(MPITestCase):
    N_PROCS=4

    def setUp(self):
        self.startdir = os.getcwd()

        if self.comm.rank == 0:
            self.tmpdir = mkdtemp()
        else:
            self.tmpdir = None

        #make sure everyone is using the same temp directory
        self.tmpdir = self.comm.bcast(self.tmpdir, root=0)

        os.chdir(self.tmpdir)

    def tearDown(self):
        os.chdir(self.startdir)
        if self.comm.rank == 0:
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
        sink = prob.root.add("sink", FileSink('sink', self.N_PROCS))

        for i in range(self.N_PROCS):
            par.add("mid%d"%i, FileMid('mid%d'%i,'mid%d'%i))
            prob.root.connect('src.fout', 'par.mid%d.fin'%i)
            prob.root.connect('par.mid%d.fout'%i, 'sink.fin%d'%i)

        prob.setup(check=False)
        prob.run()

        for i in range(self.N_PROCS):
            with sink.params['fin%d'%i].open('r') as f:
                self.assertEqual(f.read(), "src\npar.mid%d\n"%i)

    def test_file_diamond_same_names(self):
        prob = Problem(Group(), impl=impl)

        src = prob.root.add("src", FileSrc('foo'))
        par = prob.root.add('par', ParallelGroup())
        sink = prob.root.add("sink", FileSink('sink', self.N_PROCS))

        for i in range(self.N_PROCS):
            # all FileMids will have output file with the same name, so
            # framework needs to create rank specific directories for
            # each output file to avoid collisions.
            par.add("mid%d"%i, FileMid('foo','foo'))
            prob.root.connect('src.fout', 'par.mid%d.fin'%i)
            prob.root.connect('par.mid%d.fout'%i, 'sink.fin%d'%i)

        prob.setup(check=False)
        prob.run()

        for i in range(self.N_PROCS):
            with sink.params['fin%d'%i].open('r') as f:
                self.assertEqual(f.read(), "src\npar.mid%d\n"%i)


if __name__ == '__main__':
    from openmdao.test.mpi_util import mpirun_tests
    mpirun_tests()
