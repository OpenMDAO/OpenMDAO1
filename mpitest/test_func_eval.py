
import time

from openmdao.test.mpi_util import MPITestCase
from openmdao.util.concurrent import concurrent_eval_lb

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

if MPI:
    comm = MPI.COMM_WORLD
    rank = comm.rank
    if comm.size == 1:
        # don't bother with MPI since we only have one proc
        comm = None
else:
    comm = None
    rank = 0


def funct(job, option=None):
    time.sleep(1)
    if job == 5:
        raise RuntimeError("Job 5 had an (intentional) error!")
    if MPI:
        rank = MPI.COMM_WORLD.rank
    else:
        rank = 0
    return (job, option, rank)


class MPITests(MPITestCase):

    N_PROCS = 6

    def test_simple(self):

        ncases = 10

        cases = [([i], {'option': 'foo%d'%i}) for i in range(ncases)]
        tofind = set(range(ncases))
        found = set()

        results = concurrent_eval_lb(funct, cases, comm)

        if comm is None or comm.rank == 0:
            self.assertEqual(len(results), 10)
            for r in results:
                if r[0] is not None:
                    found.add(r[0][0])
                else:
                    self.assertTrue('Job 5 had an (intentional) error!' in r[1])
            diff = tofind-found
            self.assertEqual(diff, set([5])) # job 5 should have failed
        else:
            self.assertEqual(results, None)

    def test_bcast(self):

        ncases = 10

        cases = [([i], {'option': 'foo%d'%i}) for i in range(ncases)]
        tofind = set(range(ncases))
        found = set()

        results = concurrent_eval_lb(funct, cases, comm, broadcast=True)

        self.assertEqual(len(results), 10)
        for r in results:
            if r[0] is not None:
                found.add(r[0][0])
            else:
                self.assertTrue('Job 5 had an (intentional) error!' in r[1])
        diff = tofind-found
        self.assertEqual(diff, set([5])) # job 5 should have failed



if __name__ == '__main__':
    from openmdao.test.mpi_util import mpirun_tests
    mpirun_tests()
