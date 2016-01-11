
"""
Some classes to make it a little more convenient to do MPI testing.
"""

from __future__ import print_function

import os
import sys
import time
import traceback
import subprocess
from six.moves import cStringIO as StringIO
from six import PY3
import unittest
from unittest import TestCase, SkipTest
from os.path import join, dirname, abspath
import inspect
from inspect import getmembers, ismethod, isfunction, isclass, getargspec

# in python3, inspect.ismethod doesn't work as you might expect, so...
if PY3:
    def ismethod(obj):
        return isfunction(obj) or inspect.ismethod(obj)


from openmdao.core.mpi_wrap import under_mpirun, FakeComm


try:
    from mpi4py import MPI
except ImportError:
    class MPITestCase(TestCase):
        def __init__(self, methodName='runTest'):
            super(MPITestCase, self).__init__(methodName=methodName)
            self.comm = FakeComm()

    mpirun_tests = unittest.main

else:
    class MPITestCase(TestCase):
        """A base class for all TestCases that are
        intended to run under mpirun.
        """

        # A class attribute 'N_PROCS' must be defined
        # for each MPITestCase class in order to
        # know how big to make the MPI communicator.
        # N_PROCS = 4
        def __init__(self, methodName='runTest'):
            super(MPITestCase, self).__init__(methodName=methodName)
            self.comm = MPI.COMM_WORLD

    class TestResult(object):
        """Contains the path to the test function/method, status
        of the test (if finished), error and stdout messages (if any),
        and start/end times.
        """

        def __init__(self, testspec, start_time, end_time,
                     status='OK', err_msg=''):
            self.testspec = testspec
            self.status = status
            self.err_msg = err_msg
            self.start_time = start_time
            self.end_time = end_time

        def elapsed(self):
            return self.end_time - self.start_time

        def __str__(self):
            if self.err_msg:
                return "%s: %s\n%s" % (self.testspec, self.status, self.err_msg)
            else:
                return "%s: %s" % (self.testspec, self.status)

    def try_call(method):
        """Calls the given method, captures stdout and stderr,
        and returns the status (OK, SKIP, FAIL).
        """
        status = 'OK'
        try:
            method()
        except Exception as e:
            msg = traceback.format_exc()
            if isinstance(e, SkipTest):
                status = 'SKIP'
                sys.stderr.write(str(e))
            else:
                status = 'FAIL'
                sys.stderr.write(msg)
        except:
            msg = traceback.format_exc()
            status = 'FAIL'
            sys.stderr.write(msg)

        sys.stderr.flush()

        return status

    def run_test(testspec, parent, method, nocap=False):
        start_time = time.time()

        errstream = StringIO()
        if nocap:
            outstream = sys.stdout
        else:
            outstream = StringIO()

        setup = getattr(parent, 'setUp', None)
        teardown = getattr(parent, 'tearDown', None)

        run_method = True
        run_td = True

        try:
            old_err = sys.stderr
            old_out = sys.stdout
            sys.stdout = outstream
            sys.stderr = errstream

            # if there's a setUp method, run it
            if setup:
                status = try_call(setup)
                if status != 'OK':
                    run_method = False
                    run_td = False

            if run_method:
                status = try_call(getattr(parent, method))

            if teardown and run_td:
                tdstatus = try_call(teardown)
                if status == 'OK':
                    status = tdstatus

            result = TestResult(testspec, start_time, time.time(), status,
                                errstream.getvalue())

        finally:
            sys.stderr = old_err
            sys.stdout = old_out

        return result

    def mpirun_tests(args=None):
        """This is used in the "if __name__ == '__main__'" block to run all
        tests in that module.  The tests are run using mpirun.
        """
        if args is None:
            args = sys.argv[1:]

        mod = __import__('__main__')

        if '-ns' in args:
            nocap = False
        else:
            nocap = True

        tests = [n for n in args if not n.startswith('-')]
        options = [n for n in args if n.startswith('-')]

        if tests:
            for test in tests:
                tcase, _, method = test.partition('.')
                if method:
                    parent = getattr(mod, tcase)(methodName=method)
                    if hasattr(parent, 'N_PROCS') and not under_mpirun():
                        retcode = run_in_sub(getattr(mod, tcase), test, options)
                        continue
                else:
                    raise NotImplementedError("module test functions not supported yet")
                    #parent = mod
                    #method = tcase

                tspec = "%s:%s" % (mod.__file__, test)
                if MPI.COMM_WORLD.rank == 0:
                    sys.stdout.write("%s ... " % tspec)
                result = run_test(tspec, parent, method, nocap=nocap)

                if under_mpirun():
                    results = MPI.COMM_WORLD.gather(result, root=0)

                    if MPI.COMM_WORLD.rank == 0:
                        for i,r in enumerate(results):
                            if r.status != 'OK':
                                sys.stdout.write("%s\nERROR in rank %d:\n" % (r.status, i))
                                sys.stdout.write("%s\n" % r.err_msg)
                                break
                        else:
                            sys.stdout.write("%s\n" % r.status)
                else:
                    sys.stdout.write("%s\n" % r.status)

        else: # find all test methods in the file and mpi run ourselves for each one

            for k,v in getmembers(mod, isclass):
                if issubclass(v, TestCase):
                    for n, method in getmembers(v, ismethod):
                        if n.startswith('test_'):
                            testspec = k+'.'+n
                            if not hasattr(v, 'N_PROCS'):
                                print(run_test(testspec, v(methodName=n), n, nocap=nocap))
                            else:
                                retcode = run_in_sub(v, testspec, options)


def run_in_sub(testcase, testspec, options):
    mod = __import__('__main__')

    from distutils import spawn
    mpirun_exe = None
    if spawn.find_executable("mpirun") is not None:
        mpirun_exe = "mpirun"
    elif spawn.find_executable("mpiexec") is not None:
        mpirun_exe = "mpiexec"

    if mpirun_exe is None:
        raise Exception("mpirun or mpiexec was not found in the system path.")


    cmd = "%s -n %d %s %s %s %s" % \
               (mpirun_exe, testcase.N_PROCS, sys.executable,
                mod.__file__, testspec, ' '.join(options))
    
    return subprocess.call(cmd, shell=True)
