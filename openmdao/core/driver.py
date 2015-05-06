""" Base class for Driver."""


class Driver(object):
    """ Base class for drivers in OpenMDAO. Drivers can only be placed in a
    Problem, and every problem has a Driver. Driver is the simplest driver that
    runs (solves using solve_nonlinder) a problem once.
    """

    def add_param(self, name, low=None, high=None):
        pass

    def add_objective(self, name):
        pass

    def add_constraint(self, name):
        pass

    def run(self, system):
        """ Runs the driver. This function should be overriden when inheriting.

        system: system
            System that our parent Problem owns.
        """

        varmanager = system._varmanager
        params = varmanager.params
        unknowns = varmanager.unknowns
        resids = varmanager.resids

        system.solve_nonlinear(params, unknowns, resids)

