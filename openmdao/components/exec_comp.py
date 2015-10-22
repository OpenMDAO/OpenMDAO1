""" Class definition for ExecComp, a component that evaluates an expression."""

import math
import cmath

import numpy
from numpy import ndarray, complex, imag

from six import string_types

from openmdao.core.component import Component
from openmdao.util.string_util import parse_for_vars
from openmdao.util.array_util import array_idx_iter


class ExecComp(Component):
    """
    Given a list of assignment statements, this component creates
    input and output variables at construction time.  All variables
    appearing on the left-hand side of an assignment are outputs,
    and the rest are inputs.  Each variable is assumed to be of
    type float unless the initial value for that variable is supplied
    in \*\*kwargs.  Derivatives are calculated using complex step.

    Args
    ----
    exprs: str or list of str
        An assignment statement or iter of them. These express how the
        outputs are calculated based on the inputs.

    \*\*kwargs: dict of named args
        Initial values of variables can be set by setting a named
        arg with the var name.

    Options
    -------
    fd_options['force_fd'] :  bool(False)
        Set to True to finite difference this system.
    fd_options['form'] :  str('forward')
        Finite difference mode. (forward, backward, central) You can also set to 'complex_step' to peform the complex step method if your components support it.
    fd_options['step_size'] :  float(1e-06)
        Default finite difference stepsize
    fd_options['step_type'] :  str('absolute')
        Set to absolute, relative

    Notes
    -----
    In order to create an ExecComp with array variables, or any other
    variable type that is not a float, you must use a keyword arg to
    set the initial value of each non-float variable.  For example,
    let's say we have an ExecComp that takes an array 'x' as input and
    outputs a float variable 'y' which is the sum of the entries in 'x'.

    >>> import numpy
    >>> from openmdao.api import ExecComp
    >>> excomp = ExecComp('y=numpy.sum(x)', x=numpy.ones(10,dtype=float))

    In this example, 'y' would be assumed to be the default type of float
    and would be given an initial value of 0.0, while 'x' would be
    initialized with a size 100 float array.
    """

    def __init__(self, exprs, **kwargs):
        super(ExecComp, self).__init__()

        # if complex step is used for derivatives, this is the stepsize
        self.complex_stepsize = 1.e-6

        if isinstance(exprs, string_types):
            exprs = [exprs]

        self._codes = [compile(expr, expr, 'exec') for expr in exprs]

        outs = set()
        allvars = set()

        # find all of the variables and which ones are outputs
        for expr in exprs:
            lhs, _ = expr.split('=', 1)
            outs.update(parse_for_vars(lhs))
            allvars.update(parse_for_vars(expr, kwargs.keys()))

        # make sure all kwargs are legit
        for kwarg in kwargs:
            if kwarg not in allvars:
                raise RuntimeError("Keyword arg '%s' in call to ExecComp() "
                                   "does not refer to any variable in the "
                                   "expressions %s" % (kwarg, exprs))

        for var in sorted(allvars):
            # if user supplied an initial value, use it, otherwise set to 0.0
            val = kwargs.get(var, 0.0)

            if var in outs:
                self.add_output(var, val)
            else:
                self.add_param(var, val)


    def solve_nonlinear(self, params, unknowns, resids):
        """
        Executes this component's assignment statemens.

        Args
        ----
        params : `VecWrapper`, optional
            `VecWrapper` containing parameters. (p)

        unknowns : `VecWrapper`, optional
            `VecWrapper` containing outputs and states. (u)

        resids : `VecWrapper`, optional
            `VecWrapper` containing residuals. (r)
        """
        for expr in self._codes:
            exec(expr, _expr_dict, _UPDict(unknowns, params))

    def linearize(self, params, unknowns, resids):
        """
        Uses complex step method to calculate a Jacobian dict.

        Args
        ----
        params : `VecWrapper`
            `VecWrapper` containing parameters. (p)

        unknowns : `VecWrapper`
            `VecWrapper` containing outputs and states. (u)

        resids : `VecWrapper`
            `VecWrapper` containing residuals. (r)

        Returns
        -------
        dict
            Dictionary whose keys are tuples of the form ('unknown', 'param')
            and whose values are ndarrays.
        """

        # our complex step
        step = self.complex_stepsize * 1j

        J = {}

        for param in params:

            pwrap = _TmpDict(params)

            pval = params[param]
            if isinstance(pval, ndarray):
                # replace the param array with a complex copy
                pwrap[param] = numpy.asarray(params[param], complex)
                idx_iter = array_idx_iter(pwrap[param].shape)
                psize = pval.size
            else:
                pwrap[param] = complex(params[param])
                idx_iter = (None,)
                psize = 1

            for i, idx in enumerate(idx_iter):
                # set a complex param value
                if idx is None:
                    pwrap[param] += step
                else:
                    pwrap[param][idx] += step

                uwrap = _TmpDict(unknowns, complex=True)

                # solve with complex param value
                self.solve_nonlinear(pwrap, uwrap, resids)

                for u in unknowns:
                    jval = imag(uwrap[u] / self.complex_stepsize)
                    if (u, param) not in J: # create the dict entry
                        J[(u, param)] = numpy.zeros((jval.size, psize))

                    # set the column in the Jacobian entry
                    J[(u, param)][:, i] = jval.flat

                # restore old param value
                if idx is None:
                    pwrap[param] -= step
                else:
                    pwrap[param][idx] -= step

        return J


class _TmpDict(object):
    """
    A wrapper for a dictionary that will allow getting of values
    from its inner dict unless those values get modified via
    __setitem__.  After values have been modified they are managed
    thereafter by the wrapper.  This protects the inner dict from
    modification.

    Args
    ----
    inner : dict-like
        The dictionary to be wrapped.

    complex : bool, optional
        If True, return a complex version of values from __getitem__
    """
    def __init__(self, inner, complex=False):
        self._inner = inner
        self._changed = {}
        self._complex = complex

    def __getitem__(self, name):
        if name in self._changed:
            return self._changed[name]
        elif self._complex:
            val = self._inner[name]
            if isinstance(val, ndarray):
                self._changed[name] = numpy.asarray(val, dtype=complex)
            else:
                self._changed[name] = complex(val)
            return self._changed[name]
        else:
            return self._inner[name]

    def __setitem__(self, name, value):
        self._changed[name] = value

    def __contains__(self, name):
        return name in self._inner or name in self._changed

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _UPDict(object):
    """
    A dict-like wrapper for the unknowns and params
    objects.  Items are first looked for in the unknowns
    and then the params.

    Args
    ----
    unknowns : dict-like
        The unknowns object to be wrapped.

    params : dict-like
        The params object to be wrapped.
    """
    def __init__(self, unknowns, params):
        self._unknowns = unknowns
        self._params = params

    def __getitem__(self, name):
        try:
            return self._unknowns[name]
        except KeyError:
            return self._params[name]

    def __setitem__(self, name, value):
        if name in self._unknowns:
            self._unknowns[name] = value
        elif name in self._params:
            self._params[name] = value
        else:
            self._unknowns[name] # will raise KeyError


def _import_functs(mod, dct, names=None):
    """
    Maps attributes attrs from the given module into the given dict.

    Args
    ----
    dct : dict
        Dictionary that will contain the mapping

    names : iter of str, optional
        If supplied, only map attrs that match the given names
    """
    if names is None:
        names = dir(mod)
    for name in names:
        if isinstance(name, tuple):
            name, alias = name
        else:
            alias = name
        if not name[0] == '_':
            dct[name] = getattr(mod, name)
            dct[alias] = dct[name]


# this dict will act as the local scope when we eval our expressions
_expr_dict = {}

# Note: no function in the math module supports complex args, so the following can only be used
#       in ExecComps if derivatives are not required.  The functions below don't have numpy
#       versions (which do support complex args), otherwise we'd just use those.  Some of these
#       will be overridden if scipy is found.
_import_functs(math, _expr_dict,
               names=['factorial', 'fsum', 'lgamma', 'erf', 'erfc', 'gamma'])

_import_functs(numpy, _expr_dict,
               names=['cosh', 'ldexp', 'hypot', 'tan', 'isnan', 'log', 'fabs',
                      'floor', 'sqrt', 'frexp', 'degrees', 'pi', 'log10', 'modf',
                      'copysign', 'cos', 'ceil', 'isinf', 'sinh', 'trunc',
                      'expm1', 'e', 'tanh', 'radians', 'sin', 'fmod', 'exp', 'log1p',
                      ('arcsin', 'asin'), ('arcsinh', 'asinh'), ('arctanh', 'atanh'),
                      ('arctan', 'atan'), ('arctan2', 'atan2'),
                      ('arccosh', 'acosh'), ('arccos', 'acos'),
                      ('power', 'pow')])

# Note: adding cmath here in case someone wants to have an ExecComp that
# performs some complex operation during solve_nonlinear. cmath functions
# generally return complex numbers even if the args are floats.
_expr_dict['cmath'] = cmath

_expr_dict['numpy'] = numpy


# if scipy is available, add some functions
try:
    import scipy.special
except ImportError:
    pass
else:
    _import_functs(scipy.special, _expr_dict, names=['gamma', 'polygamma', 'erf', 'erfc'])
