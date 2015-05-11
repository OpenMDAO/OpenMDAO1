"""Utilities for the OpenMDAO test process."""

from math import isnan


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

