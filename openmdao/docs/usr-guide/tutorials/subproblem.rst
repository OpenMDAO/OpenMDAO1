.. _OpenMDAO-Subproblem:


Subproblem Tutorial -  Running Multiple Optimizations Using SubProblems
=======================================================================

In this tutorial, we want to find the global minimum of a function that has
multiple local minima, and we want to search for those local minima using
multiple gradient based optimizers running concurrently.  How might we solve
this problem in OpenMDAO?  If we didn't care about concurrency, we could just
write a script that creates a single Problem containing a gradient optimizer
and the function we want to optimize, and have that script iterate over a list
of design inputs, set the design values into the Problem, run it, and extract
the objective values.  If we want to run multiple
optimizations concurrently, it turns out that OpenMDAO has a number of drivers,
for example CaseDriver, LatinHypercubeDriver, UniformDriver, etc., that will
run multiple input cases concurrently.  But how can we use multiple drivers
during an OpenMDAO run?  To do that, we need to have multiple Problems, because
in OpenMDAO, only a Problem can have a driver.

OpenMDAO has a component called `SubProblem`, which is a component that
contains a Problem and controls which of the Problem's variables are accessible
from outside.  We'll use one of those to contain the Problem that performs
a gradient based optimization using an SLSQP optimizer, and we'll add that to
our top level Problem, which will run multiple instances of our SubProblem
concurrently using a CaseDriver.


.. note::

    There is some overhead involved in using a SubProblem, so using one is
    not recommended unless your approach truly requires nested drivers.  Some
    valid uses of SubProblem would be:

        - collaborative optimization
        - an optimizer on top of a DOE
        - a DOE on top of an optimizer, a.k.a. multistart optimization  (our case)
        - a genetic algorithm driving a gradient based optimizer


Let's first create a Problem to contain the optimization of our function.
Later, we'll use this Problem to create our SubProblem.

.. testcode:: subprob

    import sys
    from math import pi

    from openmdao.api import Problem, Group, Component, IndepVarComp, ExecComp, \
                             ScipyOptimizer, SubProblem, CaseDriver


    sub = Problem(root=Group())
    root = sub.root


Now let's define the function we want to minimize.  In this case
we've chosen a simple function with only one input and one output.
It's a cosine function between the bounds += pi that is modified so that
the rightmost "valley" is slightly lower than valleys to the left.  Between
the += pi bounds, there are only two valleys, so we have two local minima and
one of those is global.

The code below defines a component that represents our function, as well as
an independent variable that the optimizer can use as a design variable. We
put both of those in the root Group and connect our independent variable to our
component's input.


.. testcode:: subprob

    # In the range  -pi <= x <= pi
    # function has 2 local minima, one is global
    #
    # global min is:  f(x) = -1.31415926 at x = pi
    # local min at: f(x) = -0.69084489952  at x = -3.041593

    # define the independent variable that our optimizer will twiddle
    root.add('indep', IndepVarComp('x', 0.0))

    # here's the actual function we're minimizing
    root.add("comp", ExecComp("fx = cos(x)-x/10."))

    # connect the independent variable to the input of our function component
    root.connect("indep.x", "comp.x")


Now we'll set up our SLSQP optimizer.  We first declare our optimizer object,
then add our independent variable `indep.x` to it as a design variable,
then finally add the output of our component, `comp.fx`, as the objective that
we want to minimize.

.. testcode:: subprob

    sub.driver = ScipyOptimizer()
    sub.driver.options['optimizer'] = 'SLSQP'

    sub.driver.add_desvar("indep.x", lower=-pi, upper=pi)

    sub.driver.add_objective("comp.fx")

.. testcode:: subprob
    :hide:

    sub.driver.options['disp'] = False  # disable optimizer output

The lower level Problem is now completely defined.  Next we'll create the
top level Problem that will contain our SubProblem.  Also, and this is a little
confusing, we add an independent variable `top_indep.x` to the root of our
top level Problem, even though we already have an independent variable that
will feed our function inside of our lower level Problem. We need to do this
because an OpenMDAO driver can only set its design values into variables
belonging to an IndepVarComp, and the IndepVarComp in the SubProblem is not
accessible to the driver in the top level Problem.

.. testcode:: subprob

    prob = Problem(root=Group())

    prob.root.add("top_indep", IndepVarComp('x', 0.0))


Now we create our SubProblem, exposing `indep.x` as a parameter and `comp.fx`
as an unknown.  `indep.x` must be a parameter on our SubProblem in order for
us to connect our top level independent variable `top_indep.x` to it.  It's
OK that `indep.x` is in fact an unknown inside of our SubProblem.


.. testcode:: subprob

    prob.root.add("subprob", SubProblem(sub, params=['indep.x'],
                                        unknowns=['comp.fx']))

    prob.root.connect("top_indep.x", "subprob.indep.x")


Next we specify our top level driver to be a CaseDriver, which is a driver
that will execute a user defined list of cases on the model.  A case is just
a list of (name, value) tuples, where `name` is the name of a design variable
and `value` is the value that will be assigned to that variable prior to
running the model.  We're using a CaseDriver here for simplicity, and because
we already know where the local minima are found, but we could just as easily
use a LatinHyperCubeDriver that would give us some random distribution of
starting points in the design space.

Because the function we're minimizing in this tutorial has only two local
minima, we'll create our CaseDriver with an argument of `num_par_doe=2`,
specifying that we want to run 2 cases concurrently.  We'll also add
`top_indep.x` as a design variable to our CaseDriver, and add `subprob.indep.x`
and `subprob.comp.fx` as response variables.  `add_response()` is telling our
CaseDriver that we want it to save the specified variables each time it runs
an input case.  Note that `add_response()` is just a convenience method and
results in the creation of a memory resident data recorder in the CaseDriver.


.. note::

    If you want to run lots of cases and/or the variables you want to record are
    large, you may want to use some other form of data recorder,
    e.g., SqliteRecorder, to record results to disk rather than storing them
    all in memory by using add_response().  Recorders can be added to a
    CaseDriver in the same way as for any other driver.


.. code-block:: python

    prob.driver = CaseDriver(num_par_doe=2)

    prob.driver.add_desvar('top_indep.x')
    prob.driver.add_response(['subprob.indep.x', 'subprob.comp.fx'])


.. testcode:: subprob
    :hide:

    import sys
    if sys.platform == 'win32':
        prob.driver = CaseDriver(num_par_doe=1)
    else:
        prob.driver = CaseDriver(num_par_doe=2)

    prob.driver.add_desvar('top_indep.x')
    prob.driver.add_response(['subprob.indep.x', 'subprob.comp.fx'])


Next we'll define the cases we want to run. The top_indep.x values of
-1 and 1 will end up at the local and global minima when we run the concurrent
subproblem optimizers.


.. testcode:: subprob

    prob.driver.cases = [
        [('top_indep.x', -1.0)],
        [('top_indep.x',  1.0)]
    ]


Finally, we setup and run the top level problem.  Calling run() on the problem
will run the concurrent optimizations.


.. testcode:: subprob

    prob.setup(check=False)
    prob.run()


After running, we can collect the responses from our CaseDriver and the response
with the minimum value of `subprob.comp.fx` will give us our global minimum.


.. testcode:: subprob

    optvals = []

    # collect responses for all of our input cases
    optvals = [dict(resp) for resp, success, msg in prob.driver.get_responses()]

    # find the minimum value of subprob.comp.fx in our responses
    global_opt = sorted(optvals, key=lambda x: x['subprob.comp.fx'])[0]
    print("\nGlobal optimum:\nsubprob.comp.fx = %s  at  subprob.indep.x = %s" %
          (global_opt['subprob.comp.fx'], global_opt['subprob.indep.x']))


.. testoutput:: subprob
    :options: +ELLIPSIS, +NORMALIZE_WHITESPACE
    :hide:


    Global optimum:
    subprob.comp.fx = -1.31415...  at  subprob.indep.x = 3.14159...



.. note::

   If we were trying to minimize a function where we didn't know all of the
   local minima ahead of time, there would be no guarantee that this approach
   would locate all of them, and therefore no guarantee that the minimum of
   our local minima would be the actual global minimum.


Putting it all together, it looks like this:


.. code-block:: python

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


    if __name__ == '__main__':
        # First, define a Problem to be able to optimize our function.
        sub = Problem(root=MultiMinGroup())

        # set up our SLSQP optimizer
        sub.driver = ScipyOptimizer()
        sub.driver.options['optimizer'] = 'SLSQP'
        sub.driver.options['disp'] = False  # disable optimizer output

        # In this case, our design variable is indep.x, which happens
        # to be connected to the x parameter on our 'comp' component.
        sub.driver.add_desvar("indep.x", lower=-pi, upper=pi)

        # We are minimizing comp.fx, so that's our objective.
        sub.driver.add_objective("comp.fx")


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
        # separate optimizations concurrently.  This time around we'll
        # just run 2 concurrent cases.
        prob.driver = CaseDriver(num_par_doe=2)

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
        print("\nGlobal optimum:\n  subprob.comp.fx = %s   at  subprob.indep.x = %s" %
              (global_opt['subprob.comp.fx'], global_opt['subprob.indep.x']))


Output
------

::

    Global optimum:
    subprob.comp.fx = -1.31415926536   at  subprob.indep.x = 3.14159265359

.. tags:: Tutorials, SubProblem, Problem
