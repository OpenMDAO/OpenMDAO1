""" Base class for Driver."""

from collections import OrderedDict

import numpy as np

from openmdao.core.options import OptionsDictionary


class Driver(object):
    """ Base class for drivers in OpenMDAO. Drivers can only be placed in a
    Problem, and every problem has a Driver. Driver is the simplest driver that
    runs (solves using solve_nonlinear) a problem once.
    """

    def __init__(self):
        super(Driver, self).__init__()
        self.recorders = []
        self._outputs_of_interest = []
        self._inputs_of_interest = []

        # What this driver supports
        self.supports = OptionsDictionary(read_only=True)
        self.supports.add_option('Inequality Constraints', True)
        self.supports.add_option('Equality Constraints', True)
        self.supports.add_option('Linear Constraints', False)
        self.supports.add_option('Multiple Objectives', False)
        self.supports.add_option('2-Sided Constraints', False)
        self.supports.add_option('Integer Parameters', False)

        # This driver's options
        self.options = OptionsDictionary()

        self._params = OrderedDict()
        self._objs = OrderedDict()
        self._cons = OrderedDict()

        # We take root during setup
        self.root = None

    def _setup(self, root):
        """ Prepares some things we need."""
        self.root = root

        item_names = ['Parameter', 'Objective', 'Constraint']
        items = [self._params, self._objs, self._cons]

        for item, item_name in zip(items, item_names):
            for name, meta in item.items():

                # Check validity of variable
                if name not in root.unknowns:
                    msg = "{} '{}' not found in unknowns."
                    msg = msg.format(item_name, name)
                    raise ValueError(msg)

                # Size is useful metadata to save
                meta['size'] = root.unknowns.metadata(name)['size']

    def add_recorder(self, recorder):
        self.recorders.append(recorder)

    def add_param(self, name, low=None, high=None):
        """ Adds a param to this driver.

        Parameters
        ----------
        name : string
           Name of the paramcomp in the root system.

        low : float or ndarray (optional)
            Lower boundary for the param

        high : upper or ndarray (optional)
            Lower boundary for the param
        """

        if low is None:
            low = -1e99

        if high is None:
            high = 1e99

        # TODO: Check validity of param string.
        # TODO: Check validity of everything else.

        if isinstance(low, np.ndarray):
            low = low.flat

        if isinstance(high, np.ndarray):
            high = high.flat

        param = {}
        param['low'] = low
        param['high'] = high

        self._params[name] = param

    def get_params(self):
        """ Returns a dict of parameters.

        Returns
        -------
        dict
            Keys are the param object names, and the values are the param
            values.
        """
        uvec = self.root.unknowns
        params = OrderedDict()

        for key, val in self._params.items():
            params[key] = uvec.flat[key]

        return params

    def get_param_metadata(self):
        """ Returns a dict of parameter metadata.

        Returns
        -------
        dict
            Keys are the param object names, and the values are the param
            values.
        """
        return self._params

    def set_param(self, name, value):
        """ Sets a parameter.

        Parameters
        ----------
        name : string
           Name of the paramcomp in the root system.

        val : ndarray or float
            value to set the parameter
        """
        self.root.unknowns[name] = value

    def add_objective(self, name):
        """ Adds an objective to this driver.

        Parameters
        ----------
        name : string
            Promoted pathname of the output that will serve as the objective.
        """

        # TODO: Check validity of input.

        obj = {}
        self._objs[name] = obj

    def get_objectives(self, return_type='dict'):
        """ Adds a constraint to this driver.

        Parameters
        ----------
        return_type : string
            Set to 'dict' to return a dictionary, or set to 'array' to return a
            flat ndarray.

        Returns
        -------
        dict (for return_type 'dict')
            Key is the constraint name string, value is an ndarray with the values.

        ndarray (for return_type 'array')
            Array containing all constraint values in the order they were added.
        """
        uvec = self.root.unknowns
        objs = OrderedDict()

        for key, val in self._objs.items():
            objs[key] = uvec.flat[key]

        return objs

    def add_constraint(self, name, ctype='ineq', linear=False, jacs=None):
        """ Adds a constraint to this driver.

        Parameters
        ----------
        name : string
            Promoted pathname of the output that will serve as the objective.

        ctype : string
            Set to 'ineq' for inequality constraints, or 'eq' for equality
            constraints. Make sure your driver supports the ctype of constraint
            that you are adding.

        linear : bool, optional
            Set to True if this constraint is linear with respect to all params
            so that it can be calculated once and cached.

        jacs : dict of functions, optional
            Dictionary of user-defined functions that return the flattened
            Jacobian of this constraint with repsect to the params of
            this driver, as indicated by the dictionary keys. Default is None
            to let OpenMDAO calculate all derivatives.
        """

        # TODO: Check validity of input.

        con = {}
        con['linear'] = linear
        con['ctype'] = ctype
        self._cons[name] = con

    def get_constraints(self, ctype='all', lintype='all', return_type='dict'):
        """ Gets all constraints for this driver.

        Parameters
        ----------
        ctype : string
            Default is 'all'. Optionally return just the inequality constraints
            with 'ineq' or the equality constraints with 'eq'.

        lintype : string
            Default is 'all'. Optionally return just the linear constraints
            with 'linear' or the nonlinear constraints with 'nonlinear'.

        return_type : string
            Set to 'dict' to return a dictionary, or set to 'array' to return a
            flat ndarray.

        Returns
        -------
        dict (for return_type 'dict')
            Key is the constraint name string, value is an ndarray with the values.

        ndarray (for return_type 'array')
            Array containing all constraint values in the order they were added.
        """
        uvec = self.root.unknowns
        cons = OrderedDict()

        for key, val in self._cons.items():

            if lintype=='linear' and val['linear']==False:
                continue

            if lintype=='nonlinear' and val['linear']==True:
                continue

            if ctype=='eq' and val['ctype']=='ineq':
                continue

            if ctype=='ineq' and val['ctype']=='eq':
                continue

            cons[key] = uvec.flat[key]

        return cons

    def get_constraint_metadata(self):
        """ Returns a dict of constraint metadata.

        Returns
        -------
        dict
            Keys are the constraint object names, and the values are the param
            values.
        """
        return self._cons

    def run(self, problem):
        """ Runs the driver. This function should be overriden when inheriting.

        Parameters
        ----------
        problem : `Problem`
            Our parent `Problem`.
        """
        system = problem.root
        system.solve_nonlinear()
        for recorder in self.recorders:
            recorder._record(system.params, system.unknowns, system.resids)
