.. _Design-Doc:

============
Design
============

Overview
--------

[First we need a high-level discussion of the relationships between Problem,
System, Group, Driver, Component, etc.  This should NOT mention how things used
to be.  This doc is only concerned with the present.]

System
------

Base class for systems in OpenMDAO.

This class represents a system of equations, which is a set of equations that
need to be solved together so that a single solution satisfies them all.

A system of equations is specified by one or more parameters (input values) and
one or more unknowns (output values). For example, a really simple system may
contain only a single equation, such as:

    y = x + 2

For this system, *x* is the parameter (input variable) and *y* is the unknown
(output variable). The `System` class has a *parameters* attribute and an *unknowns*
attribute that store the lists of parameters and unknowns as vectors for efficient
processing.

The equations themselves are encapsulated in a member function of the `System`
class called *solve_nonlinear*. Calling this user-provided function with the
parameters and unknowns vectors should compute the unknown values for the
given parameter values and put those values into the unknowns vector.

There are a few other attributes and member functions in the `System` interface,
mostly related to calculating derivatives of the equations, but this is the
essential base abstraction. There are two subclasses of `System` that are used
to actually build a model of a system of equations.  They are the `Component`
class and the `Group` class.


Component
---------

The `Component` class is used to instantiate a `System` by declaring the
parameters and unknowns and the solve_nonlinear method.  The user will extend
the `Component` class to define a system of interest. In OpenMDAO, a
`Component` is normally used to encapsulate a specific discipline or subset
of a problem.

When defining a `Component`, the user must declare the parameter and unknown
variables and define a *solve_nonlinear* function that calculates the
values of the unknowns for a given set of parameter values.

Variables are added to the system in the constructor (__init__ method) via the
*add_parameter*, *add_output* and *add_state* functions. For example:
::
    class MySystem(Component):
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
unknowns vector, and a residuals vector. It is expected that this function will
be called using the vector attributes of the containing `System`, so those vectors
will contain entries for the variables declared in the constructor. For example:
::
        def solve_nonlinear(self, params, unknowns, resids):
            unknowns['y'] = params['x'] + 2

Group
------

A complex system may be thought of as a number of coupled subsystems, which may
be represented as individual `Components` or groups of `Components`.  A `Group`
is a subclass of `System` that used to encapsulate groupings of `Systems`.

A `Group` is created simply by adding one or more `Components`, for example:
::
    group1 = Group()
    a = my_group.add(MySystem())
    b = my_group.add(MySystem())

The `Systems` in a `Group` may be either `Components` or other `Groups`. For
example, we can add a previously created `Group` to a new `Group` along with
other `Components`:
::
    group2 = Group()
    group2.add(group1)
    group2.add(MySystem())


Problem
-------

Is always the top object for running an OpenMDAO model.


Driver
------

Driver is the base class for drivers in OpenMDAO, it is the simplest driver possible,
running a problem once. Drivers can only be placed in a
Problem, and every problem has a Driver.  By default, a driver solves using solve_nonlinear.



[perhaps we could make a few diagrams to show relationships?]
