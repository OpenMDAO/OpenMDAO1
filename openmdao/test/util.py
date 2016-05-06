"""Utilities for the OpenMDAO test process."""

import os
import tempfile
import shutil

from six import iteritems

from math import isnan
import numpy as np



def problem_derivatives_check(unittest, problem, tol = 1e-5):
    """Runs partial derivates check on an OpenMDAO problem instance.
    Asserts that forward and reverse derivatives are within a specified
    relative tolerance.

    Args:
    -----
    unittest : `unittest.TestCase`
        Unit test instance.

    problem : `Problem`
        OpenMDAO problem instance to be tested.

    tol : `Float`
        Tolerance for relative error in the derivative checks.
    """
    partials = problem.check_partial_derivatives(out_stream=None)
    for comp in partials:

        derivs = partials[comp]
        for deriv in derivs.keys():
            absol = derivs[deriv]['abs error']
            err = derivs[deriv]['rel error']

            if max(absol) > 0: # zero abs error implies rel error = nan
                try:
                    unittest.assertLessEqual(max(err), tol)
                    # print "Deriv test passed:", comp, deriv, max(err)
                except AssertionError as e:
                    print("Deriv test failed:", comp, deriv, max(err))

                    raise e

def assert_rel_error(test_case, actual, desired, tolerance):
    """
    Determine that the relative error between `actual` and `desired`
    is within `tolerance`. If `desired` is zero, then use absolute error.

    Args
    ----
    test_case : :class:`unittest.TestCase`
        TestCase instance used for assertions.

    actual : float
        The value from the test.

    desired : float
        The value expected.

    tolerance : float
        Maximum relative error ``(actual - desired) / desired``.
    """
    try:
        actual[0]
    except (TypeError, IndexError):
        if isnan(actual) and not isnan(desired):
            test_case.fail('actual nan, desired %s, rel error nan, tolerance %s'
                           % (desired, tolerance))
        if desired != 0:
            error = (actual - desired) / desired
        else:
            error = actual
        if abs(error) > tolerance:
            test_case.fail('actual %s, desired %s, rel error %s, tolerance %s'
                           % (actual, desired, error, tolerance))
    else: #array values
        if not np.all(np.isnan(actual)==np.isnan(desired)):
            test_case.fail('actual and desired values have non-matching nan values')

        if np.linalg.norm(desired) == 0:
            error = np.linalg.norm(actual)
        else:
            error = np.linalg.norm(actual - desired) / np.linalg.norm(desired)

        if abs(error) > tolerance:
            test_case.fail('arrays do not match, rel error %.3e > tol (%.3e)'  % (error, tolerance))

    return error


def assert_equal_jacobian(test_case, computed_jac, expected_jac, tolerance):
    """
    Compare two jacobians, in dict format, one derivative at a time and
    check for accuracy.

    Parameters
    ----------
    test_case : :class:`unittest.TestCase`
        TestCase instance used for assertions.

    computed_jac : dict
        Computed value of the jacobian that you wish to check for accuracy.

    expected_jac : dict
        Expected jacobian values, usually computed by finite difference or
        complex step.

    tolerance : float
        Maximum relative error ``norm(computed - expected) / norm(expected)``.

    """

    computed_up_set = set(computed_jac.keys())
    expected_up_set = set(expected_jac.keys())

    for up_pair in expected_jac:
        try:
            computed_jac[up_pair]
        except KeyError:
            test_case.fail('deriv "%s" in second jacobian, but not in first' % str(up_pair))


    for up_pair, computed in iteritems(computed_jac):
        try:
            expected = expected_jac[up_pair]
        except KeyError:
            test_case.fail('deriv "%s" in first jacobian, but not in second' % str(up_pair))

        rel_err = np.linalg.norm(computed - expected)/np.linalg.norm(expected)
        abs_err = np.linalg.norm(computed - expected)

        err = min(rel_err, abs_err)

        if err > tolerance:
            test_case.fail('error for %s is %.3e, is larger than'
                'tolerance %.3e' % (str(up_pair), err, tolerance))


def assert_no_force_fd(group):
    """ Traverses the given group recursively.  If any subsystems are found
    where `fd_options['force_fd'] = True`, an AssertionError is raised.

    Parameters
    ----------
    group : OpenMDAO Group
        The system which is recursively checked for the use of
        `fd_options["force_fd"]=True`

    Raises
    ------
    AssertionError
        If a subsystem of group is found to be using
        `fd_options["force_fd"]=True`
    """
    subs = [s.pathname for s in group.subsystems(recurse=True, include_self=True)
            if s.fd_options['force_fd']]
    assert not subs, "One or more systems are using " \
                     "fd_options['force_fd']=True: " + str(subs)


def set_pyoptsparse_opt(optname):
    """For testing, sets the pyoptsparse optimizer using the given optimizer
    name.  This may be modified based on the value of
    OPENMDAO_FORCE_PYOPTSPARSE_OPT.  This can be used on systems that have
    SNOPT installed to force them to use SLSQP in order to mimic our test
    machines on travis and appveyor.
    """

    OPT = None
    OPTIMIZER = None
    force = os.environ.get('OPENMDAO_FORCE_PYOPTSPARSE_OPT')
    if force:
        optname = force

    try:
        from pyoptsparse import OPT
        try:
            OPT(optname)
            OPTIMIZER = optname
        except:
            if optname != 'SLSQP':
                try:
                    OPT('SLSQP')
                    OPTIMIZER = 'SLSQP'
                except:
                    pass
    except:
        pass

    return OPT, OPTIMIZER

class ConcurrentTestCaseMixin(object):
    def concurrent_setUp(self, prefix=''):
        """Sets up a temp dir to execute a test in so that our test cases
        can run concurrently without interfering with each other's
        input/output files.

        Args
        ----

        prefix : str, optional
            Temp directory will have this prefix.

        """
        self.startdir = os.getcwd()
        self.tempdir = tempfile.mkdtemp(prefix=prefix)
        os.chdir(self.tempdir)

    def concurrent_tearDown(self):
        os.chdir(self.startdir)
        if not os.environ.get('OPENMDAO_KEEPDIRS', False):
            try:
                shutil.rmtree(self.tempdir)
            except OSError:
                pass
