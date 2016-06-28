""" ConstraintComp is now deprecated."""

import warnings

from openmdao.components.exec_comp import ExecComp


class ConstraintComp(ExecComp):
    """

    ConstraintComp is deprecated. Please see the basic tutorial for more information.

    A Component that represents an equality or inequality constraint.

    Args
    ----
    expr : str
        Constraint expression containing an operator that is
        one of ['<', '>', '<=', '>=', '='].

    out : str, optional
        Name of the output variable containing the result of the
        constraint equation.  Default is 'out'.

    Options
    -------
    deriv_options['type'] :  str('user')
        Derivative calculation type ('user', 'fd', 'cs')
        Default is 'user', where derivative is calculated from
        user-supplied derivatives. Set to 'fd' to finite difference
        this system. Set to 'cs' to perform the complex step
        if your components support it.
    deriv_options['form'] :  str('forward')
        Finite difference mode. (forward, backward, central)
    deriv_options['step_size'] :  float(1e-06)
        Default finite difference stepsize
    deriv_options['step_calc'] :  str('absolute')
        Set to absolute, relative
    deriv_options['check_type'] :  str('fd')
        Type of derivative check for check_partial_derivatives. Set
        to 'fd' to finite difference this system. Set to
        'cs' to perform the complex step method if
        your components support it.
    deriv_options['check_form'] :  str('forward')
        Finite difference mode: ("forward", "backward", "central")
        During check_partial_derivatives, the difference form that is used
        for the check.
    deriv_options['check_step_calc'] : str('absolute',)
        Set to 'absolute' or 'relative'. Default finite difference
        step calculation for the finite difference check in check_partial_derivatives.
    deriv_options['check_step_size'] :  float(1e-06)
        Default finite difference stepsize for the finite difference check
        in check_partial_derivatives"
    deriv_options['linearize'] : bool(False)
        Set to True if you want linearize to be called even though you are using FD.
    """

    def __init__(self, expr, out='out'):

        warnings.simplefilter('always', DeprecationWarning)
        warnings.warn("ConstraintComp is deprecated, see the new add_constraint interface.",
                      DeprecationWarning,stacklevel=2)
        warnings.simplefilter('ignore', DeprecationWarning)

        newexpr = _combined_expr(expr)
        super(ConstraintComp, self).__init__("%s = %s" % (out, newexpr))


def _combined_expr(expr):
    """Given a constraint object, take the lhs, operator, and
    rhs and combine them into a single expression by moving rhs
    terms over to the lhs.  For example,
    for the constraint 'C1.x < C2.y + 7', return the expression
    'C1.x - C2.y - 7'.  Depending on the direction of the operator,
    the sign of the expression may be flipped.  The final form of
    the constraint, when evaluated, will be considered to be satisfied
    if it evaluates to a value <= 0.
    """
    lhs, op, rhs = _parse_constraint(expr)

    first, second = (rhs, lhs) if op.startswith('>') else (lhs, rhs)

    try:
        if float(first) == 0:
            return "-(%s)" % second
    except Exception:
        pass

    try:
        if float(second) == 0.:
            return first
    except Exception:
        pass

    return '%s-(%s)' % (first, second)


def _parse_constraint(expr_string):
    """ Parses the constraint expression string and returns the lhs string,
    the rhs string, and comparator
    """
    for comparator in ['==', '>=', '<=', '>', '<', '=']:
        parts = expr_string.split(comparator)
        if len(parts) == 2:
            # check for == because otherwise they get a cryptic error msg
            if comparator == '==':
                break
            return (parts[0].strip(), comparator, parts[1].strip())
        # elif len(parts) == 3:
        #     return (parts[1].strip(), comparator,
        #             (parts[0].strip(), parts[2].strip()))

    msg = "Constraints require an explicit comparator (=, <, >, <=, or >=)"
    raise ValueError(msg)
