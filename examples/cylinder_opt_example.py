"""
Example 5 from: https://cims.nyu.edu/~kiryl/Calculus/Section_4.5--Optimization%20Problems/Optimization_Problems.pdf

A manufacturer needs to make a cylindrical can that will hold 1.5 liters of
liquid. Determine the dimensions of the can that will minimize the amount
of material used in its construction.

Minimize:     area = 2*pi*r**2 + 2*pi*r*h
Constraint: volume = pi*r**2*h = 1500

Optimal values:  r = 6.2035,   h = 12.407

"""

from openmdao.api import Problem, Group, Component, IndepVarComp, ExecComp, ScipyOptimizer
from math import pi


class Cylinder(Component):
    def __init__(self):
        super(Cylinder, self).__init__()
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


class CylinderGroup(Group):
    def __init__(self):
        super(CylinderGroup, self).__init__()

        # In OpenMDAO, design variables have to be unknowns, so if you have
        # parameters that you want to drive with an optimizer like we do in
        # this case, you have to create an IndepVarComp with unknowns that can
        # be connected to the parameters that you want to drive.
        self.add("indep", IndepVarComp([('r', 1.0, {'units':'cm'}),
                                        ('h', 1.0, {'units':'cm'})]))
        self.add('cylinder', Cylinder())

        # connect our IndepVarComp unknowns to our Cylinder params
        self.connect("indep.r", "cylinder.r")
        self.connect("indep.h", "cylinder.h")


class CylinderGroupWithExecComp(Group):
    """
    This Group is the same as CylinderGroup above except that the
    cylinder is modeled using an ExecComp instead of defining a new
    class.
    """
    def __init__(self):
        super(CylinderGroupWithExecComp, self).__init__()

        self.add("indep", IndepVarComp([('r', 1.0, {'units':'cm'}),
                                        ('h', 1.0, {'units':'cm'})]))

        # using an ExecComp, we specify the same equations as seen in
        # our Cylinder class, but it computes derivatives for us
        # automatically using complex step so we don't have to define
        # our derivatives.
        self.add("cylinder", ExecComp(["area = 2.0*pi*r**2+2.0*pi*r*h",
                                       "volume = pi*r**2*h/1000."],
                                       units={'r':'cm','h':'cm',
                                              'volume':'L','area':'cm**2'}))
        self.connect("indep.r", "cylinder.r")
        self.connect("indep.h", "cylinder.h")


if __name__ == '__main__':
    # First, solve the problem using our Cylinder class (which is found inside
    # of CylinderGroup)
    prob = Problem(root=CylinderGroup())

    # set up our SLSQP optimizer
    prob.driver = ScipyOptimizer()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['disp'] = False  # don't display optimizer output

    # In this case, our design variables are indep.r and indep.h, which happen
    # to be connected to the r and h parameters on our cylinder component.
    prob.driver.add_desvar("indep.r", lower=0.0, upper=1.e99)
    prob.driver.add_desvar("indep.h", lower=0.0, upper=1.e99)

    # We are maximizing cylinder.area, so that's our objective.
    prob.driver.add_objective("cylinder.area")

    # Finally, our cylinder volume is constrained to be 1.5 liters, so we
    # add a constraint on cyclinder.volume and specify that it must equal 1.5
    prob.driver.add_constraint("cylinder.volume", equals=1.5)

    prob.setup(check=False)
    prob.run()

    # optimum should be:  indep.r = 6.2035,  indep.h = 12.407

    print("\nOptimal values using the Cylinder class:")
    for indep in ('indep.r', 'indep.h', 'cylinder.area', 'cylinder.volume'):
        print("%s = %f" % (indep, prob[indep]))


    # now solve the same problem using an ExecComp instead of our
    # Cylinder class
    prob.root = CylinderGroupWithExecComp()
    prob.setup(check=False)
    prob.run()

    print("\nOptimal values using an ExecComp:")
    for indep in ('indep.r', 'indep.h', 'cylinder.area', 'cylinder.volume'):
        print("%s = %f" % (indep, prob[indep]))
