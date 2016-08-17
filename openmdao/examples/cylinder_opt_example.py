"""
Example 5 from: https://cims.nyu.edu/~kiryl/Calculus/Section_4.5--Optimization%20Problems/Optimization_Problems.pdf

A manufacturer needs to make a cylindrical can that will hold 1.5 liters of
liquid. Determine the dimensions of the can that will minimize the amount
of material used in its construction.

Minimize:     area = 2*pi*r**2 + 2*pi*r*h
Constraint: volume = pi*r**2*h/1000 = 1.5

Optimal values:
    indep.r = 6.2035 cm
    indep.h = 12.407 cm
    cylinder.area = 725.396379 cm**2
    cylinder.volume = 1.5 L

We're going to model our cylinder in two different ways and show that
we get the same result.
"""

from openmdao.api import Problem, Group, Component, IndepVarComp, ExecComp, ScipyOptimizer
from math import pi


class Cylinder1(Component):
    """
    This is one way to model our cylinder.  We can define our own
    Component class that defines solve_nonlinear() and linearize()
    methods.
    """
    def __init__(self):
        super(Cylinder1, self).__init__()
        self.add_param('r', 0.0, units='cm', desc='radius')
        self.add_param('h', 0.0, units='cm', desc='height')
        self.add_output('area', 0.0, units='cm**2')
        self.add_output('volume', 0.0, units='L')

    def solve_nonlinear(self, params, unknowns, resids):

        r = params['r']
        h = params['h']

        unknowns['area'] = 2.0*pi*r**2+2.0*pi*r*h
        unknowns['volume'] = pi*r**2*h/1000.  # div by 1000 to convert to liters

    def linearize(self, params, unknowns, resids):
        J = {}

        r = params['r']
        h = params['h']

        J['area','r'] = 4.0*pi*r + 2.0*pi*h
        J['area','h'] = 2.0*pi*r
        J['volume','r'] = 2.0*pi*r*h/1000.
        J['volume','h'] = pi*r**2/1000.

        return J


def Cylinder2():
    """
    This is another way to model our cylinder, by using an ExecComp
    that contains the equations for cylinder area and volume
    given radius and height.
    """
    return ExecComp(["area = 2.0*pi*r**2+2.0*pi*r*h",
                     "volume = pi*r**2*h/1000."],
                     units={'r':'cm','h':'cm',
                            'volume':'L','area':'cm**2'})


def setup_opt(cylinder):
    """
    Creates a Problem, adds a model for the cylinder, and sets up the
    optimizer, design variables, and constraints.
    """
    prob = Problem(root=Group())
    root = prob.root

    # add our cylinder component and our independent variables to the
    # root Group

    # In OpenMDAO, design variables have to be unknowns, so if you have
    # parameters that you want to drive with an optimizer like we do in
    # this case, you have to create an IndepVarComp with unknowns that can
    # be connected to the parameters that you want to drive.
    root.add("indep", IndepVarComp([('r', 1.0, {'units':'cm'}),
                                    ('h', 1.0, {'units':'cm'})]))
    root.add('cylinder', cylinder)

    # connect our independent variables to our cylinder params
    root.connect("indep.r", "cylinder.r")
    root.connect("indep.h", "cylinder.h")

    # set up our SLSQP optimizer
    prob.driver = ScipyOptimizer()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['disp'] = False  # don't display optimizer output

    # In this case, our design variables are indep.r and indep.h, which happen
    # to be connected to the r and h parameters on our cylinder component.
    prob.driver.add_desvar("indep.r", lower=0.0, upper=1.e99)
    prob.driver.add_desvar("indep.h", lower=0.0, upper=1.e99)

    # We are minimizing cylinder.area, so that's our objective.
    prob.driver.add_objective("cylinder.area")

    # Finally, our cylinder volume is constrained to be 1.5 liters, so we
    # add a constraint on cyclinder.volume and specify that it must equal 1.5
    prob.driver.add_constraint("cylinder.volume", equals=1.5)

    return prob


def opt_cylinder1():
    """
    Perform the optimization using our Cylinder class.
    """
    prob = setup_opt(Cylinder1())

    prob.setup(check=False)
    prob.run()

    vnames = ('indep.r', 'indep.h', 'cylinder.area', 'cylinder.volume')

    # return the optimal values for all cylinder variables
    return [(name, prob[name]) for name in vnames]


def opt_cylinder2():
    """
    Perform the optimization using an ExecComp to model the cylinder.
    """
    prob = setup_opt(Cylinder2())

    prob.setup(check=False)
    prob.run()

    vnames = ('indep.r', 'indep.h', 'cylinder.area', 'cylinder.volume')

    # return the optimal values for all cylinder variables
    return [(name, prob[name]) for name in vnames]



if __name__ == '__main__':

    # optimal values should be:
    #     indep.r = 6.2035
    #     indep.h = 12.407
    #     cylinder.area = 725.396379
    #     cylinder.volume = 1.5

    print("\nOptimal values using the Cylinder class:")
    for name, val in opt_cylinder1():
        print("%s = %f" % (name, val))

    print("\nOptimal values using an ExecComp:")
    for name, val in opt_cylinder2():
        print("%s = %f" % (name, val))
