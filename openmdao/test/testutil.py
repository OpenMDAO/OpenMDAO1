"""Utilities for the OpenMDAO test process."""

from six import iteritems

from math import isnan
import numpy as np

def assert_rel_error(test_case, actual, desired, tolerance):
    """
    Determine that the relative error between `actual` and `desired`
    is within `tolerance`. If `desired` is zero, then use absolute error.

    Parameters
    ----------
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
    else:
        for i, (act, des) in enumerate(zip(actual, desired)):
            if isnan(act) and not isnan(des):
                test_case.fail('at %d: actual nan, desired %s, rel error nan,'
                               ' tolerance %s' % (i, des, tolerance))
            if des != 0:
                error = (act - des) / des
            else:
                error = act
            if abs(error) > tolerance:
                test_case.fail('at %d: actual %s, desired %s, rel error %s,'
                               ' tolerance %s' % (i, act, des, error, tolerance))


def assert_equal_jacobian(test_case, computed_jac, expected_jac, tolerance):
    """
    Compare two jacobians, in dict format, one derivative at a time and
    check for accuracy.

    Parameters
    ----------
    test_case : :class:`unittest.TestCase`
        TestCase instance used for assertions.

    computed_jac : dict
        Computed value of the jacobian that you wish to check for accuracy

    expected_jac : dict
        Expected jacobian values, usually computed by finite difference or
        complex step

    tolerance : float
        Maximum relative error ``norm(computed - expected) / norm(expected)``.

    """

    for unknown, computed in iteritems(computed_jac):
        try:
            expected = expected_jac[unknown]
        except KeyError:
            test_case.fail('unknown "%s" in first jacobian, but not in second' % unknown)

        for param, J in iteritems(computed):
            try:
                J_expected = expected[param]
            except KeyError:
                test_case.fail('param "%s", for unknown "%s", in first jacobian, but not in second' % (param, unknown))

            rel_err = np.linalg.norm(J - J_expected)/np.linalg.norm(J_expected)
            abs_err = np.linalg.norm(J - J_expected)

            err = min(rel_err, abs_err)

            if err > tolerance:
                test_case.fail('error for d_%s/d_%s is %.3e, is larger than'
                    'tolerance %.3e' % (unknown, param, err, tolerance))
