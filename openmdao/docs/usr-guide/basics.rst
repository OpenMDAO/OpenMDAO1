
.. _Basics:

======
Basics
======


Overview
--------

This document is a brief overview of the core concepts and constructs involved in
defining and solving a problem using OpenMDAO.


System
------

The base class for building complex system models in OpenMDAO is the `System`
class. This class represents a system of equations, which is a set of equations
that need to be solved together so that a single solution satisfies them all.

A system of equations is specified by one or more parameters (input values) and
one or more unknowns (output values). For example, a really simple system may
contain only a single equation, such as:

    y = x + 2

For this system, *x* is the parameter (input variable) and *y* is the unknown
(output variable). The `System` class has a *parameters* attribute and an *unknowns*
attribute that store the lists of parameters and unknowns as vectors for efficient
processing.

The equations themselves for a simple system like this are encapsulated in a
member function of the `System` class called *solve_nonlinear*. Calling this
user-provided function with the parameters and unknowns vectors should compute
the unknown values for the given parameter values and put those values into
the unknowns vector.

There are a few other attributes and member functions in the `System` interface,
mostly related to calculating derivatives and supporting more complex systems,
but this is the essential base abstraction.

There are two subclasses of `System` that are used to actually build a model
of a system of equations.  They are the `Component` class and the `Group` class.


Component
---------

The `Component` class is used to instantiate a `System` by declaring the
parameters and unknowns and the solve_nonlinear method. The user will extend
the `Component` class to define a system of interest. In OpenMDAO, a
`Component` is normally used to encapsulate a specific discipline or subset
of a problem.

When defining a `Component`, the user must declare the parameter and unknown
variables and define a *solve_nonlinear* function that calculates the
values of the unknowns for a given set of parameter values.

Variables are added to the system in the constructor (__init__ method) via the
*add_parameter*, *add_output* and *add_state* functions. For example:

::

    class MyComp(Component):
        def __init__(self):
            self.add_param('x', val=0.)

            self.add_output('y', val=0.)

            self.add_state('z', val=[0., 1.])

Note that unknowns come in two flavors: *outputs* that represent the explicit
output values for the equations, and *states* which represent an internal state
of the system. Initial values are required when adding variables to a system
in order to specify the type and size/shape of the variable, needed to allocate
space in the corresponding vectors.

The *solve_nonlinear* function takes three arguments: the parameters vector, the
unknowns vector, and a residuals vector. This function will be called using the
vector attributes of the containing `System`, so those vectors will contain entries
for the variables declared in the constructor. For example:

::

        def solve_nonlinear(self, params, unknowns, resids):
            unknowns['y'] = params['x'] + 2

The user may optionally provide a *jacobian* method that computes the derivatives
of the unknowns with respect to the parameters. This function is needed when
using a solver that makes use of analytic derivatives (this is discussed elsewhere
in the documentation).


Group
------

A complex system may be modeled as a number of coupled subsystems, which may
be represented as individual `Components` or groups of `Components`.  A `Group`
is a subclass of `System` that used to encapsulate groupings of `Systems`.

A `Group` is created simply by adding one or more `Systems`. Those `Systems`
may be either `Components` or other `Groups`. For example, we can add a `Group`
to another `Group` along with some `Components`:

::

    c1 = MyComp()
    c2 = MyComp()
    c3 = MyComp()

    g1 = Group()
    g1.add(c1)
    g1.add(c2)

    g2 = Group()
    g2.add(c3)
    g2.add(g1)

Interdependencies between `Systems` in a `Group` are represented as connections
between the `Group`'s subsystems.  Connections can be made in two ways: explicitly
or implicitly.

An explicit connection is made from the output of one `System` to the input
(parameter) of another using the `Group` *connect* method, as follows:

::

    g1.connect('c1.y', 'c2.x')

Alternatively, you can use the *promotion* mechanism to implicitly connect two
or more variables.  When a `System` is added to a `Group`, you may optionally
specify a list of variable names that are to be *promoted* from the subsystem
to the group level. This means that you can reference the variable as if it
were an attribute of the `Group` rather than the subsystem.  For Example:

::

    g2.add(c3, promotes=['x'])

Now you can access the parameter 'x' from 'c3' as if it were an attribute of
the group: 'g2.x'. If you promote multiple subsystem variables with the same
name, then those variables will be implicitly connected:

::

    g2.add(g1, promotes=['c1.x'])

Now setting a value for 'g2.x' will set the value for both 'c3.x' and 'g1.c1.x'
and they are said to be implicitly connected.  If you promote the output from
one subsystem and the input of another with the same name, then that will have
the same effect to the explicit connection statement as shown above.

In contrast to a `Component`, which is reponsible for defining the variables
and equations of a system, a `Group` has the responsibility of assembling
multiple systems of equations into matrix form and solving them together.
Where a `Component` must define a *solve_nonlinear* method, a `Group` provides
a solver to solve the collection of `Components` as a whole. In fact, a `Group`
has two associated solvers: a linear solver and a non-linear solver.  The
default linear solver is SciPy's GMres and the default non-linear solver is a
simple `RunOnce` solver that will just call the solve_non_linear method on each
system in the `Group` sequentially. A number of other solvers, both linear and
non-linear, are available that can be substituted for the defaults for
different use cases.


Problem
-------

When a model has been fully developed as a `Group` with a collection of
`Components` and sub-`Groups` it is time to actually solve the `System`.
This is done by definining a `Problem` that contains the `System`.

A `Problem` always has a single top-level `Group` called *root*.  This can
be passed in the constructor or set later:

::

    prob = Problem(ExampleGroup())

    or

    root = ExampleGroup()
    prob = Problem(root)

A `Problem` also has a driver, which "drives" or controls the solution of
the `Problem`. The base `Driver` class in OpenMDAO is the simplest driver
possible, which just calls *solve_nonlinear* on the *root* `Group`. This
simple driver may be replaced with a different type of driver depending on the
problem to be solved.  Specifically, drivers are provided to support optimization
using the SciPy *minimize* family of local optimizers and the SNOPT optimization
software package. Examples showing how to use these optimizers can be found
elsewhere in the documentation.

The `Driver` is invoked by calling the *run* method on the `Problem`. Prior
to doing that, however, you must perform *setup*.  This function does all
the necessary initialization of the data vectors and configuration for the
data transfers that must occur during execution. An optional but highly
recommended additional step is to call the *check_setup* method after calling
*setup*. This will look for and report any potential issues with the `Problem`
configuration, including unconnected parameters, conflicting units, etc.

Summary
-------

The general procedure for defining and solving a `Problem` in OpenMDAO is:
    - define `Components` (including their *solve_nonlinear*  and optional *jacobian* functions)
    - assembling `Components` into Groups and making connections (explicitly or implicitly)
    - instantiating a `Problem` with the *root* `Group`
    - perform *setup* on the `Problem` to initialize all vectors and data transfers
    - perform *check_setup* on the `Problem` to identify any issues
    - perform *run* on the Problem

A very basic example of defining and running a `Problem` as discussed here is shown below.
This example makes use of a couple of convenience components to provide a source for the
parameter (`ParamComp`) and to quickly define a `Component` for an equation (`ExecComp`).

::

    from openmdao.core.group import Group
    from openmdao.core.problem import Problem
    from openmdao.components.paramcomp import ParamComp
    from openmdao.components.execcomp import ExecComp

    root = Group()
    root.add('x_param', ParamComp('x', 7.0))
    root.add('mycomp', ExecComp('y=x*2.0'))
    root.connect('x_param.x', 'mycomp.x')

    prob = Problem(root)
    prob.setup()
    prob.check_setup()
    prob.run()

    result = root.unknowns['mycomp.y']
