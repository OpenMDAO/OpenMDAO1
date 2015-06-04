
import re
import math
import ast

import numpy
from numpy import zeros, ndarray

from six import string_types

from openmdao.core.component import Component
from openmdao.util.strutil import parse_for_vars


class ExecComp(Component):
    """
    Given a list of assignment statements, this component creates
    input and output variables at construction time.  All variables
    appearing on the left-hand side of the assignments are outputs,
    and the rest are inputs.  Each variable is assumed to be of
    type float unless the initial value for that variable is supplied
    in **kwargs.

    Parameters
    ----------
    exprs: str or iter of str
        An assignment statement or iter of them. These express how the
        outputs are calculated based on the inputs.

    derivs: str or iter of str, optional
        An assignment statement or iter of them.  These specify how the
        derivatives are calculated.  Derivative names must be of the form
        d<var>_d<wrt> for a derivative of <var> with respect to <wrt>.

    **kwargs: dict of named args
        Initial values of variables can be set by setting a named
        arg with the var name.
    """

    def __init__(self, exprs, derivs=(), **kwargs):
        super(ExecComp, self).__init__()

        if isinstance(exprs, string_types):
            exprs = [exprs]

        self.exprs = exprs
        self.codes = [compile(expr, expr, 'exec') for expr in exprs]

        # dictionary used for the exec of the compiled assignment statements
        self._exec_dict = {}

        outs = set()
        allvars = set()

        # find all of the variables and which ones are outputs
        for expr in exprs:
            lhs, rhs = expr.split('=')
            outs.update(parse_for_vars(lhs))
            allvars.update(parse_for_vars(expr))

        for var in allvars:
            # if user supplied an initial value, use it, otherwise set to 0.0
            val = kwargs.get(var, 0.0)

            # need to initialize _exec_dict to prevent keyerror when
            # params are not connected
            self._exec_dict[var] = val

            if var in outs:
                self.add_output(var, val)
            else:
                self.add_param(var, val)

        self._setup_derivs(derivs, **kwargs)

    def _setup_derivs(self, derivs, **kwargs):
        self.deriv_exprs = derivs
        if not derivs:
            return

        if isinstance(derivs, string_types):
            derivs = [derivs]

        self.deriv_codes = \
            [compile(expr, expr, 'exec') for expr in derivs]

        allvars = set()

        self.deriv_names = []
        deriv_rgx = re.compile('d(\w+)_d(\w+)')
        for expr in derivs:
            lhs, _ = expr.split('=')
            for lhs in parse_for_vars(lhs):
                match = deriv_rgx.search(lhs)
                numerator = match.group(1)
                wrt = match.group(2)

                if numerator not in self._unknowns_dict:
                    raise RuntimeError("Derivative numerator '%s' could not be found" %
                                       numerator)
                if wrt not in self._params_dict:
                    raise RuntimeError("Derivative denominator '%s' could not be found" %
                                       wrt)

                self.deriv_names.append( (lhs, numerator, wrt) )

                self._exec_dict[lhs] = kwargs.get(lhs, 0.0)

    def solve_nonlinear(self, params, unknowns, resids):
        """
        Executes this component's assignment statements
        """
        for pname, meta in params.items():
            self._exec_dict[pname] = params[pname]

        for expr in self.codes:
            exec(expr, _expr_dict, self._exec_dict )

        for uname in unknowns.keys():
            unknowns[uname] = self._exec_dict[uname]

    def jacobian(self, params, unknowns, resids):
        """
        Calculate the Jacobian using our derivative expressions.
        """
        if not self.deriv_exprs:
            return None

        for pname, meta in params.items():
            self._exec_dict[pname] = params[pname]

        for expr in self.deriv_codes:
            exec(expr, _expr_dict, self._exec_dict )

        J = {}

        for dname, numerator, wrt in self.deriv_names:
            deriv = self._exec_dict[dname]
            if not isinstance(deriv, ndarray):
                deriv = numpy.array([deriv])
            J[(numerator, wrt)] = deriv

        return J


def _import_functs(mod, dct, names=None):
    """
    Maps attributes attrs from the given module into the given dict.

    Parameters
    ----------
    dct : dict
        Dictionary that will contain the mapping

    names : iter of str, optional
        If supplied, only map attrs that match the given names
    """
    if names is None:
        names = dir(mod)
    for name in names:
        if not name.startswith('_'):
            dct[name] = getattr(mod, name)


# this dict will act as the local scope when we eval our expressions
_expr_dict = {}


# add stuff from math lib directly to our locals dict so users won't have to
# put 'math.' in front of all of their calls to standard math functions


_import_functs(numpy, _expr_dict,
               names=['array', 'cosh', 'ldexp', 'hypot', 'tan', 'isnan', 'log', 'fabs',
                      'floor', 'sqrt', 'frexp', 'degrees', 'pi', 'log10', 'modf',
                      'copysign', 'cos', 'ceil', 'isinf', 'sinh', 'trunc',
                      'expm1', 'e', 'tanh', 'radians', 'sin', 'fmod', 'exp', 'log1p'])

_import_functs(math, _expr_dict,
               names=['asin', 'asinh', 'atanh', 'atan', 'atan2', 'factorial',
                      'fsum', 'lgamma', 'erf', 'erfc', 'acosh', 'acos', 'gamma'])

_expr_dict['math'] = math
_expr_dict['pow'] = numpy.power #pow in math is not complex stepable, but this one is!
_expr_dict['numpy'] = numpy


# if scipy is available, add some functions
try:
    import scipy.special
except ImportError:
    pass
else:
    _import_functs(scipy.special, _expr_dict, names=['gamma', 'polygamma'])


