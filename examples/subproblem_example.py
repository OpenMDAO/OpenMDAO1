
import sys

from openmdao.api import Problem, Group, Component, IndepVarComp, ExecComp, \
                         ScipyOptimizer, SubProblem, CaseDriver
from math import pi


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


if __name__ == '__main__':
    # First, define our SubProblem to be able to optimize our six hump
    # camelback function.
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

    prob.root.add("indep", IndepVarComp('x', 0.0))

    # add our subproblem
    prob.root.add("subprob", SubProblem(sub, params=['indep.x'],
                                        unknowns=['comp.fx']))

    prob.root.connect("indep.x", "subprob.indep.x")

    # use a CaseDriver as our top level driver so we can run multiple
    # separate optimizations concurrently.  In this simple case we'll
    # just run 2 concurrent cases.
    prob.driver = CaseDriver(num_par_doe=2)

    prob.driver.add_desvar('indep.x')
    prob.driver.add_response(['subprob.indep.x', 'subprob.comp.x', 'subprob.comp.fx'])

    prob.driver.cases = [
        [('indep.x', -1.0)],
        [('indep.x',  1.0)]
    ]

    prob.setup(check=False)

    prob.run()

    optvals = []
    print("\nValues found at local minima using the multi min function:")
    for i, (responses, success, msg) in enumerate(prob.driver.get_responses()):
        responses = dict(responses)
        optvals.append(responses)
        sys.stdout.write("Min %d: " % i)
        for j, (name, val) in enumerate(responses.items()):
            sys.stdout.write("%s = %s" % (name, val))
            if j==0:
                sys.stdout.write(", ")
        sys.stdout.write("\n")

    optvals = sorted(optvals, key=lambda x: x['subprob.comp.fx'])
    opt = optvals[0]
    print("\nGlobal optimum:\n  subprob.comp.fx = %s   at  subprob.indep.x = %s" %
          (opt['subprob.comp.fx'], opt['subprob.indep.x']))
