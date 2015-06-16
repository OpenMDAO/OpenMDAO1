""" Base class for Driver."""

from collections import OrderedDict

from openmdao.core.options import OptionsDictionary


class Driver(object):
    """ Base class for drivers in OpenMDAO. Drivers can only be placed in a
    Problem, and every problem has a Driver. Driver is the simplest driver that
    runs (solves using solve_nonlinear) a problem once.
    """

    def __init__(self):
        super(Driver, self).__init__()
        self.recorders = []

        self.supports = OptionsDictionary(read_only=True)
        self.supports.add_option('Inequality Constraints', True)
        self.supports.add_option('Equality Constraints', True)
        self.supports.add_option('Linear Constraints', False)
        self.supports.add_option('Multiple Objectives', True)

        self._params = OrderedDict()
        self._objs = OrderedDict()
        self._eq_cons = OrderedDict()
        self._ineq_cons = OrderedDict()

        # We take root during setup
        self.root = None

    def add_recorder(self, recorder):
        self.recorders.append(recorder)

    def add_param(self, name, low=None, high=None, scaler=None, adder=None):
        """ Adds a param to this driver.

        Parameters
        ----------
        name : string
           Name of the paramcomp in the root system.

        low : float or ndarray (optional)
            Lower boundary for the param

        high : upper or ndarray (optional)
            Lower boundary for the param

        scalar : upper or ndarray (optional)
            Multiplicative scale from physical to normalized coordinates.

        adder : upper or ndarray (optional)
            Additive scale from physical to normalized coordinates.
        """

        if low is None:
            low = -1e99

        if high is None:
            high = 1e99

        # TODO: Check validity of param string.
        # TODO: Check validity of everything else.

        param = {}
        param['low'] = low
        param['high'] = high
        param['scaler'] = scaler
        param['adder'] = adder

        self._params[name] = param

    def get_parameters(self):
        """ Returns a dict of parameters.

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
        pass

    def add_objective(self, name, scaler=None, adder=None):
        """ Adds an objective to this driver.

        Parameters
        ----------
        name : string
            Promoted pathname of the output that will serve as the objective.

        scalar : upper or ndarray (optional)
            Multiplicative scale from physical to normalized coordinates.

        adder : upper or ndarray (optional)
            Additive scale from physical to normalized coordinates.
        """
        pass

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
        pass

    def add_constraint(self, name, ctype='ineq', linear=False, scaler=None,
                       adder=None, jacs=None):
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

        scalar : upper or ndarray (optional)
            Multiplicative scale from physical to normalized coordinates.

        adder : upper or ndarray (optional)
            Additive scale from physical to normalized coordinates.

        jacs : dict of functions, optional
            Dictionary of user-defined functions that return the flattened
            Jacobian of this constraint with repsect to the params of
            this driver, as indicated by the dictionary keys. Default is None
            to let OpenMDAO calculate all derivatives.
        """
        pass

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
        pass

    def _post_setup(self):
        """ Do anything that we need to do before we run. Note, all this
        stuff could be in `run`, but we are thinking ahead to nested
        problems."""
        pass

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
