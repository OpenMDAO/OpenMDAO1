""" Base class for Driver."""


class Driver(object):
    """ Base class for drivers in OpenMDAO. Drivers can only be placed in a
    Problem, and every problem has a Driver. Driver is the simplest driver that
    runs (solves using solve_nonlinear) a problem once.
    """

    def __init__(self):
        super(Driver, self).__init__()
        self.recorders = []
        self.outputs_of_interest = []
        self.inputs_of_interest = []

    def add_param(self, name, low=None, high=None):
        pass

    def add_objective(self, name):
        pass

    def add_constraint(self, name):
        pass

    def add_recorder(self, recorder):
        self.recorders.append(recorder)

    def run(self, system):
        """ Runs the driver. This function should be overriden when inheriting.

        Parameters
        ----------
        system : `System`
            `System` that our parent `Problem` owns.
        """
        system.solve_nonlinear()
