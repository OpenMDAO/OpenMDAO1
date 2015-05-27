import os
import sys
import numpy
from contextlib import contextmanager

from six import reraise, PY3


def _redirect_streams(to_fd):
    """
    Redirect stdout/stderr to the given file descriptor.
    Based on: http://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python/
    """

    original_stdout_fd = sys.stdout.fileno()
    original_stderr_fd = sys.stderr.fileno()

    # Flush and close sys.stdout/err - also closes the file descriptors (fd)
    sys.stdout.close()
    sys.stderr.close()

    # Make original_stdout_fd point to the same file as to_fd
    os.dup2(to_fd, original_stdout_fd)
    os.dup2(to_fd, original_stderr_fd)

    # Create a new sys.stdout that points to the redirected fd
    if PY3:
        sys.stdout = io.TextIOWrapper(os.fdopen(original_stdout_fd, 'wb'))
        sys.sterr = io.TextIOWrapper(os.fdopen(original_stdout_fd, 'wb'))
    else:
        sys.stdout = os.fdopen(original_stdout_fd, 'wb', 0) # 0 makes them unbuffered
        sys.stderr = os.fdopen(original_stderr_fd, 'wb', 0)

def use_proc_files():
    """Calling this will cause stdout/stderr from each MPI process to be written
    to a separate file in the current directory named <rank>.out.
    """
    if MPI is not None:
        rank = MPI.COMM_WORLD.rank
        sname = "%s.out" % rank
        ofile = open(sname, 'wb')
        _redirect_streams(ofile.fileno())

def under_mpirun():
    """Return True if we're being executed under mpirun."""
    # this is a bit of a hack, but there appears to be
    # no consistent set of environment vars between MPI
    # implementations.
    for name in os.environ.keys():
        if name.startswith('OMPI_COMM') or \
           name.startswith('MPIR_')     or \
           name.startswith('MPICH_'):
            return True
    return False


if under_mpirun():
    from mpi4py import MPI
    def debug(msg):
        print("%d: %s" % (MPI.COMM_WORLD.rank, msg))
        sys.stderr.flush()
        sys.stdout.flush()
else:
    MPI = None
    def debug(msg):
        print(msg)

class FakeComm(object):
    def __init__(self):
        self.rank = 0
        self.size = 1


def get_comm_if_active(system, comm=None):
    """
    Return an MPI communicator or a fake communicator if not running under MPI.
    If running under MPI and current rank exceeds the max processes usable by
    the given system, COMM_NULL will be returned.

    Parameters
    ----------
    system : a `System`
        The system that is requesting a communicator.

    comm : an MPI communicator (real or fake)
        The communicator being offered by the parent system.

    Returns
    -------
    MPI communicator or a fake MPI commmunicator
    """
    if MPI:
        if comm is None or comm == MPI.COMM_NULL:
            return comm

        req, max_req = system.get_req_procs()

        # if we can use every proc in comm, then we're good
        if max_req is None or max_req >= comm.size:
            return comm

        # otherwise, we have to create a new smaller comm that
        # doesn't include the unutilized processes.
        if comm.rank+1 > max_req:
            color = MPI.UNDEFINED
        else:
            color = 1

        return comm.Split(color)
    else:
        return FakeComm()

def evenly_distrib_idxs(num_divisions, arr_size):
    """
    Given a number of divisions and the size of an array, chop the array up
    into pieces according to number of divisions, keeping the distribution
    of entries as even as possible. Returns a tuple of
    (sizes, offsets), where sizes and offsets contain values for all
    divisions.
    """
    base = arr_size / num_divisions
    leftover = arr_size % num_divisions
    sizes = numpy.ones(num_divisions, dtype="int") * base

    # evenly distribute the remainder across size-leftover procs,
    # instead of giving the whole remainder to one proc
    sizes[:leftover] += 1

    offsets = numpy.zeros(num_divisions, dtype="int")
    offsets[1:] = numpy.cumsum(sizes)[:-1]

    return sizes, offsets


@contextmanager
def MultiProcFailCheck():
    """Wrap this around code that you want to globally fail if it fails
    on any MPI process in MPI_WORLD.  If not running under MPI, don't
    handle any exceptions.
    """
    if MPI is None:
        yield
    else:
        try:
            yield
        except:
            exc_type, exc_val, exc_tb = sys.exc_info()
            if exc_val is not None:
                fail = True
            else:
                fail = False

            fails = MPI.COMM_WORLD.allgather(fail)

            if fail or not any(fails):
                six.reraise(exc_type, exc_val, exc_tb)
            else:
                for i,f in enumerate(fails):
                    if f:
                        raise RuntimeError("a test failed in (at least) rank %d" % i)


if os.environ.get('USE_PROC_FILES'):
    use_proc_files()
