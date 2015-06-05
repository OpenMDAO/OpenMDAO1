
import re
import math
import ast

import numpy
from numpy import zeros, ndarray, complex, imag

from six import string_types

from openmdao.core.component import Component
from openmdao.util.strutil import parse_for_vars
from openmdao.util.arrayutil import array_idx_iter


class ExecComp(Component):
    """
    Given a list of assignment statements, this component creates
    input and output variables at construction time.  All variables
    appearing on the left-hand side of the assignments are outputs,
    and the rest are inputs.  Each variable is assumed to be of
    type float unless the initial value for that variable is supplied
    in **kwargs.  Derivatives are calculated using complex step.

    Parameters
    ----------
    exprs: str or iter of str
        An assignment statement or iter of them. These express how the
        outputs are calculated based on the inputs.

    **kwargs: dict of named args
        Initial values of variables can be set by setting a named
        arg with the var name.
    """

    def __init__(self, exprs, derivs=(), **kwargs):
        super(ExecComp, self).__init__()

        # if complex step is used for derivatives, this is the stepsize
        self.complex_stepsize = 1.e-6

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

        step = self.complex_stepsize * 1j

        J = {}

        self._exec_dict = TmpDict(self._exec_dict)

        # create temporary complex versions of any arrays in _exec_dict
        for u in unknowns:
            uval = unknowns[u]
            if isinstance(uval, ndarray):
                self._exec_dict[u] = numpy.asarray(uval, dtype=complex)

        for param, pmeta in params.items():

            pwrap = TmpDict(params)

            pval = params[param]
            if isinstance(pval, ndarray):
                # replace the param array with a complex copy
                pwrap[param] = numpy.asarray(params[param], complex)
                idx_iter = array_idx_iter(pwrap[param].shape)
                psz = pval.size
            else:
                pwrap[param] = complex(params[param])
                idx_iter = (None,)
                psz = 1

            for i,idx in enumerate(idx_iter):
                if idx is None:
                    pwrap[param] += step
                else:
                    pwrap[param][idx] += step
                uwrap = TmpDict(unknowns)

                self.solve_nonlinear(pwrap, uwrap, resids)

                for u in unknowns:
                    jval = imag(uwrap[u] / self.complex_stepsize)
                    if (u,param) not in J: # create the dict entry
                        J[(u,param)] = numpy.zeros((jval.size, psz))
                    J[(u,param)][:,i] = jval

                # restore old param value
                if idx is None:
                    pwrap[param] -= step
                else:
                    pwrap[param][idx] -= step

        # unwrap _exec_dict
        self._exec_dict = self._exec_dict._inner

        return J


class TmpDict(object):
    """
    A wrapper for a dictionary that will allow getting of values
    from its inner dict unless those values get modified via
    __setitem__.  After values have been modified they are managed
    thereafter by the wrapper.  This protects the inner dict from
    modification.

    Parameters
    ----------
    inner, dict-like
        The dictionary to be wrapped.
    """
    def __init__(self, inner):
        self._inner = inner
        self._changed = {}

    def __getitem__(self, name):
        if name in self._changed:
            return self._changed[name]
        else:
            return self._inner[name]

    def __setitem__(self, name, value):
        self._changed[name] = value

    def __getattr__(self, name):
        return getattr(self._inner, name)


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


