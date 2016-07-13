"""
Some simple examples of optimzation problems
"""

from openmdao.api import Problem, Group, IndepVarComp, ExecComp, ScipyOptimizer

# The following example is from:
#
# https://cims.nyu.edu/~kiryl/Calculus/Section_4.5--Optimization%20Problems/Optimization_Problems.pdf
#

def OptimizeCylinder():
    """
    Example 5:
    A manufacturer needs to make a cylindrical can that will hold 1.5 liters of
    liquid. Determine the dimensions of the can that will minimize the amount
    of material used in its construction.

    Minimize:     area = 2*pi*r**2 + 2*pi*r*h
    Constraint: volume = pi*r**2*h = 1500

    Optimal values:  r = 6.2035,   h = 12.407

    Returns:
      the model Group,
      a driver that may be used to optimize the Group,
      a list of design vars and their optimal values in (name, opt_val) tuples
    """

    root = Group()
    root.add("indep", IndepVarComp([('r', 1.0), ('h', 1.0)]))
    root.add("cylinder", ExecComp(["area = 2.0*pi*r**2+2.0*pi*r*h",
                                   "volume = pi*r**2*h"]))
    root.connect("indep.r", "cylinder.r")
    root.connect("indep.h", "cylinder.h")

    driver = ScipyOptimizer()
    driver.options['optimizer'] = 'SLSQP'
    driver.options['disp'] = False

    driver.add_desvar("indep.r", lower=0.0, upper=1.e99)
    driver.add_desvar("indep.h", lower=0.0, upper=1.e99)
    driver.add_objective("cylinder.area")
    driver.add_constraint("cylinder.volume", equals=1500.)

    return root, driver, [('indep.r', 6.2035), ('indep.h', 12.407)]
