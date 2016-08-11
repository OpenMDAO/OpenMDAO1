
import sys
from math import pi

from openmdao.api import Problem, Group, Component, IndepVarComp, ExecComp, \
                         ScipyOptimizer, SubProblem, CaseDriver


class MultiMinGroup(Group):
    """
    In the range  -pi <= x <= pi
    function has 2 local minima, one is global

    global min is:  f(x) = -1.31415926 at x = pi
    local min at: f(x) = -0.69084489952  at x = -3.041593
    """
    def __init__(self):
        super(MultiMinGroup, self).__init__()

        self.add('indep', IndepVarComp('x', 0.0))
        self.add("comp", ExecComp("fx = cos(x)-x/10."))
        self.connect("indep.x", "comp.x")


def main(num_par_doe):
    # First, define a Problem to be able to optimize our function.
    sub = Problem(root=MultiMinGroup())

    # set up our SLSQP optimizer
    sub.driver = subdriver = ScipyOptimizer()
    subdriver.options['optimizer'] = 'SLSQP'
    subdriver.options['disp'] = False  # disable optimizer output

    # In this case, our design variable is indep.x, which happens
    # to be connected to the x parameter on our 'comp' component.
    subdriver.add_desvar("indep.x", lower=-pi, upper=pi)

    # We are minimizing comp.fx, so that's our objective.
    subdriver.add_objective("comp.fx")


    # Now, create our top level problem
    prob = Problem(root=Group())

    prob.root.add("top_indep", IndepVarComp('x', 0.0))

    # add our subproblem.  Note that 'indep.x' is actually an unknown
    # inside of the subproblem, but outside of the subproblem we're treating
    # it as a parameter.
    prob.root.add("subprob", SubProblem(sub, params=['indep.x'],
                                        unknowns=['comp.fx']))

    prob.root.connect("top_indep.x", "subprob.indep.x")

    # use a CaseDriver as our top level driver so we can run multiple
    # separate optimizations concurrently.  We'll run 'num_par_doe'
    # concurrent cases.  In this case we need no more than 2 because
    # we're only running 2 total cases.
    prob.driver = CaseDriver(num_par_doe=num_par_doe)

    prob.driver.add_desvar('top_indep.x')
    prob.driver.add_response(['subprob.indep.x', 'subprob.comp.fx'])

    # these are the two cases we're going to run.  The top_indep.x values of
    # -1 and 1 will end up at the local and global minima when we run the
    # concurrent subproblem optimizers.
    prob.driver.cases = [
        [('top_indep.x', -1.0)],
        [('top_indep.x',  1.0)]
    ]

    prob.setup(check=False)

    # run the concurrent optimizations
    prob.run()

    # collect responses for all of our input cases
    optvals = [dict(resp) for resp, success, msg in prob.driver.get_responses()]

    # find the minimum value of subprob.comp.fx in our responses
    global_opt = sorted(optvals, key=lambda x: x['subprob.comp.fx'])[0]

    return global_opt


if __name__ == '__main__':
    global_opt = main(2)

    print("\nGlobal optimum:\n  subprob.comp.fx = %s   at  subprob.indep.x = %s" %
          (global_opt['subprob.comp.fx'], global_opt['subprob.indep.x']))
