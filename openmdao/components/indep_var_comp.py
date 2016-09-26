""" OpenMDAO class definition for IndepVarComp"""

import collections
from six import string_types, iteritems

from openmdao.core.component import Component

class IndepVarComp(Component):
    """A Component that provides an output to connect to a parameter.

    """

    def __init__(self, name, val=None, **kwargs):
        super(IndepVarComp, self).__init__()

        if 'promotes' in kwargs:
            raise ValueError('IndepVarComp init: promotes is not supported in IndepVarComp.')

        if isinstance(name, string_types):
            if val is None:
                raise ValueError('IndepVarComp init: a value must be provided as the second arg.')
            self.add_output(name, val, **kwargs)

        elif isinstance(name, collections.Iterable):
            for tup in name:
                badtup = None
                if isinstance(tup, tuple):
                    if len(tup) == 3:
                        n, v, kw = tup
                    elif len(tup) == 2:
                        n, v = tup
                        kw = {}
                    else:
                        badtup = tup
                else:
                    badtup = tup
                if badtup:
                    if isinstance(badtup, string_types):
                        badtup = name
                    raise ValueError("IndepVarComp init: arg %s must be a tuple of the form "
                                     "(name, value) or (name, value, keyword_dict)." %
                                     str(badtup))
                self.add_output(n, v, **kw)
        else:
            raise ValueError("first argument to IndepVarComp init must be either of type "
                             "`str` or an iterable of tuples of the form (name, value) or "
                             "(name, value, keyword_dict).")

    def _setup_variables(self):
        """
        Returns copies of our params and unknowns dictionaries,
        re-keyed to use absolute variable names.

        """
        pdict, udict = super(IndepVarComp, self)._setup_variables()

        # mark our vars as belonging to an IndepVarComp so we can check later
        # for illegal setting of non-IndepVarComp variables.
        for name, meta in iteritems(udict):
            meta['_canset_'] = True

        return pdict, udict

    def _sys_apply_linear(self, mode, do_apply, vois=(None, ), gs_outputs=None,
                          rel_inputs=None):
        """For `IndepVarComp`, just pass on the incoming values.

        Args
        ----
        mode : string
            Derivative mode, can be 'fwd' or 'rev'.

        do_apply : dict
            We can only solve derivatives for the inputs the instigating
            system has access to. (Not used here.)

        vois: list of strings
            List of all quantities of interest to key into the mats.

        gs_outputs : dict, optional
            Linear Gauss-Siedel can limit the outputs when calling apply.

        rel_inputs : list or None (optional)
            List of inputs that are relevant for linear solve in a subsystem.
            This list only includes interior connections and states.
        """
        if mode == 'fwd':
            sol_vec, rhs_vec = self.dumat, self.drmat
            for voi in vois:
                rhs_vec[voi].vec[:] = 0.0
        else:
            sol_vec, rhs_vec = self.drmat, self.dumat

        # Note, we solve a slightly modified version of the unified
        # derivatives equations in OpenMDAO.
        # (dR/du) * (du/dr) = -I
        # The minus side on the right hand side comes from defining the
        # explicit residual to be ynew - yold instead of yold - ynew. The
        # advantage of this is that the derivative of an explicit residual is
        # the same sign as the derivative of the explicit unknown. This
        # introduces a minus one here.

        for voi in vois:
            if gs_outputs is None:
                rhs_vec[voi].vec[:] -= sol_vec[voi].vec
            else:
                for var, meta in iteritems(self.dumat[voi]):
                    if var in gs_outputs[voi]:
                        rhs_vec[voi][var] -= sol_vec[voi][var]

    def _sys_linearize(self, params, unknowns, resids, force_fd=False):
        """ No linearization needed for this one"""
        # added to avoid the small overhead from calling the parent implementation
        # because this class has nothing to do
        pass

    def solve_nonlinear(self, params, unknowns, resids):
        """ Performs no operation. """
        pass
