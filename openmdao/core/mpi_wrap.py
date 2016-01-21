""" A bunch of MPI utilities."""

import os
import sys
import io
from contextlib import contextmanager
import traceback

import six
from six import PY3


def _redirect_streams(to_fd):
    """
    Redirect stdout/stderr to the given file descriptor.
    Based on: http://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python/.
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
        sys.stderr = io.TextIOWrapper(os.fdopen(original_stdout_fd, 'wb'))
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
        if name == 'OMPI_COMM_WORLD_RANK' or \
           name == 'MPIEXEC_HOSTNAME' or \
           name.startswith('MPIR_') or \
           name.startswith('MPICH_'):
            return True
    return False


if under_mpirun():
    from mpi4py import MPI

    def debug(*msg):  # pragma: no cover
        newmsg = ["%d: " % MPI.COMM_WORLD.rank] + list(msg)
        for m in newmsg:
            sys.stdout.write("%s " % m)
        sys.stdout.write('\n')
        sys.stdout.flush()
else:
    MPI = None

    def debug(*msg):  # pragma: no cover
        for m in msg:
            sys.stdout.write("%s " % str(m))
        sys.stdout.write('\n')

class FakeComm(object):
    """ Who needs a real Comm when you have a fake one."""
    def __init__(self):
        self.rank = 0
        self.size = 1

@contextmanager
def MultiProcFailCheck():
    """ Wrap this around code that you want to globally fail if it fails
    on any MPI process in MPI_WORLD.  If not running under MPI, don't
    handle any exceptions.
    """
    if MPI is None:
        yield
    else:
        try:
            yield
        except:
            fails = MPI.COMM_WORLD.allgather(traceback.format_exc())
        else:
            fails = MPI.COMM_WORLD.allgather('')

        for i, f in enumerate(fails):
            if f:
                raise RuntimeError("a test failed in (at least) rank %d: traceback follows\n%s"
                                    % (i, f))


if os.environ.get('USE_PROC_FILES'):
    use_proc_files()
