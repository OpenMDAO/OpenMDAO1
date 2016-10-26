.. _`hohmann_tutorial`:

Hohmann Transfer Tutorial - Optimizing a Spacecraft Manuever
============================================================

This tutorial will demonstrate the use of OpenMDAO for optimizing
a simple orbital mechanics problem.  We seek the minimum possible
delta-V to transfer a spacecraft from Low Earth Orbit (LEO) to
geostationary orbit (GEO) using a two-impulse *Hohmann Transfer*.

The Hohmann Transfer is a maneuver which minimize the delta-V for
transfering a spacecraft from one circular orbit to another.  Hohmann
transfer's have a practical application in that they can be used
to transfer satellites from LEO parking orbits to geostationary orbit.

To do so, the vehicle first imparts a delta-V along the velocity vector
while in LEO.  This boosts apogee radius to the radius of the geostationary
orbit (42164 km).  In this model we will model this delta-V as an *impulsive*
maneuver which changes the spacecraft's velocity instantaneously.

We will assume that the first impulse is performed at the
ascending node in LEO.  Thus perigee of the transfer orbit is coincident
with the ascending node of the transfer orbit.  Apogee of the transfer orbit
is thus coincident with the descending node, where we will perform the
second impulse.

After the first impulse, the spacecraft coasts to apogee.  Once there
it impulse a second burn along the velocity vector to raise perigee radius
to the radius of GEO, thus circularizing the orbit.

Simple, right?  The issue is that, unless they launch from the equator,
launch vehicles do not put satellites in a low Earth parking orbit
with the same inclination as geostationary orbit.  For instance, a due east launch
from Kennedy Space Center will result in a parking orbit with an inclination of
28.5 degrees.  We therefore need to change the inclination of our satellite during
it's two impulsive burn maneuvers.  The question is, *what change in inclination
at each burn will result in the minimum possible delta-V?*

.. figure:: images/hohmann_transfer.png
   :align: center
   :alt: An inclined Hohmann Transfer diagram

   An inclined Hohmann Transfer diagram

The trajectory optimization problem can thus be stated as:

.. math::
    Minimize J=\Delta V

    s.t.

    \Delta i_1 + \Delta i_2 = 28.5

The total :math:`\Delta V` is the sum of the two impulsive :math`\Delta Vs`.  Each
of these can be determined by examining the following diagram.

[DELTA V VECTOR DIAGRAM]

The component of the delta-V in the orbital plane is along the
local horizontal plane.  The orbit-normal component is in the
direction of the desired inclination change.  Knowing the
velocity magnitude before (v1) and after (v2) the impulse, and the
change in inclination due to the impulse (\delta i), the \delta V
is then computed from the law of cosines:

.. math::
    \Delta V = v_1^2 + v_2^2 - 2 v_1 v_2 \cos{\Delta i}

In the first impulse, v_1 is the circular velocity in LEO.

.. math::
    v_c = \sqrt{\mu/r}

The velocity after the first impulse is the periapsis velocity
of the transfer orbit.  This can be solved for based on what we
know about the orbit.

The specific angular momentum of the transfer orbit is constant.
At periapsis, it is simply the product of the velocity and radius.
Therefore, rearranging we have:

.. math::
    v_p = \frac{h}{r_p}

The specific angular momentum can also be computed as:

.. math::
    h = \sqrt{p \mu}

Where p is the semilatus rectum of the orbit and \mu is
the gravitational paramter of the central body.

The semilatus rectum is computed as:


Where a and e are the semimajor axis and eccentricity of the transfer orbit, respectively.
Since we know r_a and r_p of the transfer orbit, it's semimajor axis is simply:


The eccentricity is known by the relationship of a and e to r_p (or r_a):


Thus we can compute periapsis velocity based on the periapsis and apoapsis
radii of the transfer orbit, and the gravitational parameter of the central body.

For the second impulse, the final velocity is the circular velocity of the
final orbit, which can be computed in the same way as the circular velocity
of the initial orbit.  The initial velocity at the second impulse is the
apoapsis velocity of the transfer orbit, which is:

.. math::
    v_a = \frac{h}{r_a}

Having already computed the specific angular momentum of the transfer orbit, this is
easily computed.

Finally we have the necessary calculations to compute the delta-V of the Hohmann
transfer with a plane change.

Components
----------

VCircComp
~~~~~~~~~

*VCircComp* calculates the circular orbit velocity given an orbital radius and gravitational parameter.


.. code-block:: python

    class VCircComp(Component):
        ''' Computes the circular orbit velocity given a radius and gravitational
        parameter.
        '''

        def __init__(self, radius=6378.14+400, mu=398600.4418):
            super(VCircComp, self).__init__()

            # Derivative specification (user-specified analytic derivatives)
            self.deriv_options['type'] = 'user'

            self.add_param('r', val=radius, desc='Radius from central body', units='km')
            self.add_param('mu', val=mu, desc='Gravitational parameter of central body', units='km**3/s**2')
            self.add_output('vcirc', val=1.0, desc='Circular orbit velocity at given radius and gravitational parameter', units='km/s')

        def solve_nonlinear(self, params, unknowns, resids):
            r = params['r']
            mu = params['mu']

            unknowns['vcirc'] = np.sqrt(mu/r)

        def linearize(self, params, unknowns, resids):
            r = params['r']
            mu = params['mu']
            vcirc = unknowns['vcirc']

            J = {}
            J['vcirc','mu'] = 0.5/(r*vcirc)
            J['vcirc','r'] = -0.5*mu/(vcirc*r**2)
            return J

TransferOrbitComp
~~~~~~~~~~~~~~~~~

.. code-block:: python

    class TransferOrbitComp(Component):

        def __init__(self):
            super(TransferOrbitComp, self).__init__()

            # Derivative specification
            self.deriv_options['type'] = 'fd'

            self.add_param('mu', val=398600.4418, desc='Gravitational parameter of central body', units='km**3/s**2')
            self.add_param('rp', val=7000.0, desc='periapsis radius', units='km')
            self.add_param('ra', val=42164.0, desc='apoapsis radius', units='km')

            self.add_output('vp', val=0.0, desc='periapsis velocity', units='km/s')
            self.add_output('va', val=0.0, desc='apoapsis velocity', units='km/s')

        def solve_nonlinear(self, params, unknowns, resids):

            mu = params['mu']
            rp = params['rp']
            ra = params['ra']

            a = (ra+rp)/2.0
            e = (a-rp)/a
            p = a*(1.0-e**2)

            h = np.sqrt(mu*p)

            unknowns['vp'] = h/rp
            unknowns['va'] = h/ra


DeltaVComp
~~~~~~~~~~

.. code-block:: python

    class DeltaVComp(Component):

        def __init__(self):
            super(DeltaVComp, self).__init__()

            # Derivative specification
            self.deriv_options['type'] = 'user'


            self.add_param('v1', val=1.0, desc='Initial velocity', units='km/s')
            self.add_param('v2', val=1.0, desc='Final velocity', units='km/s')
            self.add_param('dinc', val=1.0, desc='Plane change', units='rad')

            # Note:  We're going to use trigonometric functions on dinc.  The
            # automatic unit conversion in OpenMDAO comes in handy here.

            self.add_output('delta_v', val=0.0, desc='Delta-V', units='km/s')

        def solve_nonlinear(self, params, unknowns, resids):

            v1 = params['v1']
            v2 = params['v2']
            dinc = params['dinc']

            unknowns['delta_v'] = v1**2 + v2**2 - 2*v1*v2*np.cos(dinc)


        def linearize(self, params, unknowns, resids):
            v1 = params['v1']
            v2 = params['v2']
            dinc = params['dinc']

            J = {}
            J['delta_v','v1'] = 2*v1 - 2*v2*np.cos(dinc)
            J['delta_v','v2'] =  2*v2 - 2*v1*np.cos(dinc)
            J['delta_v','dinc'] = 2*v1*v2*np.sin(dinc)

            return J


Assembling the Problem
----------------------

.. code-block:: python

    prob = Problem(root=Group())

    root = prob.root

    root.add('mu_comp',IndepVarComp('mu', val=0.0,units='km**3/s**2'), promotes=['mu'])

    root.add('r1_comp',IndepVarComp('r1', val=0.0,units='km'), promotes=['r1'])
    root.add('r2_comp',IndepVarComp('r2', val=0.0,units='km'), promotes=['r2'])

    root.add('dinc1_comp', IndepVarComp('dinc1', val=0.0, units='deg'), promotes=['dinc1'])
    root.add('dinc2_comp', IndepVarComp('dinc2', val=0.0, units='deg'), promotes=['dinc2'])

    root.add('leo', system=VCircComp())
    root.add('geo', system=VCircComp())

    root.add('transfer', system=TransferOrbitComp())

    root.connect('r1', ['leo.r', 'transfer.rp'])
    root.connect('r2', ['geo.r', 'transfer.ra'])

    root.connect('mu', ['leo.mu', 'geo.mu', 'transfer.mu'])

    root.add('dv1', system=DeltaVComp())

    root.connect('leo.vcirc', 'dv1.v1')
    root.connect('transfer.vp', 'dv1.v2')
    root.connect('dinc1', 'dv1.dinc')

    root.add('dv2', system=DeltaVComp())

    root.connect('transfer.va', 'dv2.v1')
    root.connect('geo.vcirc', 'dv2.v2')
    root.connect('dinc2', 'dv2.dinc')

    root.add('dv_total', system=ExecComp('delta_v=dv1+dv2', units={'delta_v': 'km/s',
                                                                    'dv1': 'km/s',
                                                                    'dv2': 'km/s'}), promotes=['delta_v'])


    root.connect('dv1.delta_v', 'dv_total.dv1')
    root.connect('dv2.delta_v', 'dv_total.dv2')

    root.add('dinc_total', system=ExecComp('dinc=dinc1+dinc2', units={'dinc': 'deg',
                                                                    'dinc1': 'deg',
                                                                    'dinc2': 'deg'}), promotes=['dinc'])


    root.connect('dinc1', 'dinc_total.dinc1')
    root.connect('dinc2', 'dinc_total.dinc2')

    prob.driver = ScipyOptimizer()

    prob.driver.add_desvar('dinc1', lower=0, upper=28.5)
    prob.driver.add_desvar('dinc2', lower=0, upper=28.5)
    prob.driver.add_constraint('dinc', lower=28.5, upper=28.5, scaler=1.0)
    prob.driver.add_objective('delta_v', scaler=1.0)

    # Setup the problem

    prob.setup()

    # Set initial values

    prob['mu'] = 398600.4418
    prob['r1'] = 6778.137
    prob['r2'] = 42164.0

    prob['dinc1'] = 0.0
    prob['dinc2'] = 0.0

    # Go!

    prob.run()




------------ CUT -----------

This tutorial will show you how to set up a simple optimization of a paraboloid.
You'll create a paraboloid `Component` (with analytic derivatives), then put it
into a `Problem` and set up an optimizer `Driver` to minimize an objective function.

Here is the code that defines the paraboloid and then runs it. You can copy
this code into a file, and run it directly.

.. testcode:: parab

    from __future__ import print_function

    from openmdao.api import IndepVarComp, Component, Problem, Group

    class Paraboloid(Component):
        """ Evaluates the equation f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3 """

        def __init__(self):
            super(Paraboloid, self).__init__()

            self.add_param('x', val=0.0)
            self.add_param('y', val=0.0)

            self.add_output('f_xy', shape=1)

        def solve_nonlinear(self, params, unknowns, resids):
            """f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3
            """

            x = params['x']
            y = params['y']

            unknowns['f_xy'] = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0

        def linearize(self, params, unknowns, resids):
            """ Jacobian for our paraboloid."""

            x = params['x']
            y = params['y']
            J = {}

            J['f_xy', 'x'] = 2.0*x - 6.0 + y
            J['f_xy', 'y'] = 2.0*y + 8.0 + x
            return J

    if __name__ == "__main__":

        top = Problem()

        root = top.root = Group()

        root.add('p1', IndepVarComp('x', 3.0))
        root.add('p2', IndepVarComp('y', -4.0))
        root.add('p', Paraboloid())

        root.connect('p1.x', 'p.x')
        root.connect('p2.y', 'p.y')

        top.setup()
        top.run()

        print(top['p.f_xy'])


Now we will go through each section and explain how this code works.

Building the component
----------------------

::

    from __future__ import print_function

    from openmdao.api import IndepVarComp, Component, Problem, Group

We need to import some OpenMDAO classes. We also import the print_function to
ensure compatibility between Python 2.x and 3.x. You don't need the import if
you are running in Python 3.x.

::

    class Paraboloid(Component):

OpenMDAO provides a base class, `Component`, which you should inherit from to build
your own components and wrappers for analysis codes. `Components` can declare
three kinds of variables, *parameters*, *outputs* and *states*. A `Component`
operates on its parameters to compute unknowns, which can be explicit
outputs or implicit states. For the `Paraboloid` `Component`, we will only be
using explicit outputs.

::

        def __init__(self):
            super(Paraboloid, self).__init__()

            self.add_param('x', val=0.0)
            self.add_param('y', val=0.0)

            self.add_output('f_xy', shape=1)


This code defines the input parameters of the `Component`, `x` and `y`, and
initializes them to 0.0. These will be design variables which could be used to
minimize the output when doing optimization. It also defines the explicit
output, `f_xy`, but only gives it a shape. If shape is 1, the value is
initialized to *0.0*, a scalar.  If shape is any other value, the value
of the variable is initialized to *numpy.zeros(shape, dtype=float)*.

::

        def solve_nonlinear(self, params, unknowns, resids):
            """f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3
            Optimal solution (minimum): x = 6.6667; y = -7.3333
            """

            x = params['x']
            y = params['y']

            unknowns['f_xy'] = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0

The `solve_nonlinear` method is responsible for calculating outputs for a
given set of parameters. The parameters are given in the `params` dictionary
that is passed in to this method. Similarly, the outputs are assigned values
using the `unknowns` dictionary that is passed in.

::

        def linearize(self, params, unknowns, resids):
            """ Jacobian for our paraboloid."""

            x = params['x']
            y = params['y']
            J = {}

            J['f_xy','x'] = 2.0*x - 6.0 + y
            J['f_xy','y'] = 2.0*y + 8.0 + x
            return J

The `linearize` method is used to compute analytic partial derivatives of the
`unknowns` with respect to `params` (partial derivatives in OpenMDAO context refer to
derivatives for a single component by itself). The returned value, in this case `J`,
should be a dictionary whose keys are tuples of the form (‘unknown’, ‘param’) and
whose values are n-d arrays or scalars. Just like for `solve_nonlinear`, the values for the
parameters are accessed using dictionary arguments to the function.

The definition of the Paraboloid Component class is now complete. We will now
make use of this class to run a model.

Setting up the model
--------------------

::

    if __name__ == "__main__":

        top = Problem()
        root = top.root = Group()

An instance of an OpenMDAO `Problem` is always the top object for running a
model. Each `Problem` in OpenMDAO must contain a root `Group`. A `Group` is a
`System` that contains other `Components` or `Groups`.

This code instantiates a `Problem` object and sets the root to be an empty `Group`.

::

    root.add('p1', IndepVarComp('x', 3.0))
    root.add('p2', IndepVarComp('y', -4.0))

Now it is time to add components to the empty group. `IndepVarComp`
is a `Component` that provides the source for a variable which we can later give
to a `Driver` as a design variable to control.

We created two `IndepVarComps` (one for each param on the `Paraboloid`
component), gave them names, and added them to the root `Group`. The `add`
method takes a name as the first argument, and a `Component` instance as the
second argument.  The numbers 3.0 and -4.0 are values chosen for each as starting points
for the optimizer.

.. note:: Take care setting the initial values, as in some cases, various initial points for the optimization will lead to different results.


::

    root.add('p', Paraboloid())

Then we add the paraboloid using the same syntax as before, giving it the name 'p'.

::

    root.connect('p1.x', 'p.x')
    root.connect('p2.y', 'p.y')

Then we connect up the outputs of the `IndepVarComps` to the parameters of the
`Paraboloid`. Notice the dotted naming convention used to refer to variables.
So, for example, `p1` represents the first `IndepVarComp` that we created to set
the value of `x` and so we connect that to parameter `x` of the `Paraboloid`.
Since the `Paraboloid` is named `p` and has a parameter
`x`, it is referred to as `p.x` in the call to the `connect` method.

Every problem has a `Driver` and for most situations, we would want to set a
`Driver` for the `Problem` using code like this

::

    top.driver = SomeDriver()

For this very simple tutorial, we do not need to set a `Driver`, we will just
use the default, built-in driver, which is
`Driver`. ( `Driver` also serves as the base class for all `Drivers`. )
`Driver` is the simplest driver possible, running a `Problem` once.

::

    top.setup()

Before we can run our model we need to do some setup. This is done using the
`setup` method on the `Problem`. This method performs all the setup of vector
storage, data transfer, etc., necessary to perform calculations. Calling
`setup` is required before running the model.

::

    top.run()

Now we can run the model using the `run` method of `Problem`.

::

    print(top['p.f_xy'])

Finally, we print the output of the `Paraboloid` Component using the
dictionary-style method of accessing variables on the problem instance.
Putting it all together:

.. testcode:: parab

    top = Problem()
    root = top.root = Group()

    root.add('p1', IndepVarComp('x', 3.0))
    root.add('p2', IndepVarComp('y', -4.0))
    root.add('p', Paraboloid())

    root.connect('p1.x', 'p.x')
    root.connect('p2.y', 'p.y')

    top.setup()
    top.run()

    print(top['p.f_xy'])

The output should look like this:

.. testoutput:: parab
   :options: +ELLIPSIS

   -15.0

The `IndepVarComp` component is used to define a source for an unconnected
`param` that we want to use as an independent variable that can be declared as
a design variable for a driver. In our case, we want to optimize the
Paraboloid model, finding values for 'x' and 'y' that minimize the output
'f_xy.'

Sometimes we just want to run our component once to see the result.
Similarly, sometimes we have `params` that will be constant through our
optimization, and thus don't need to be design variables. In either of these
cases, the `IndepVarComp` is not required, and we can build our model while
leaving those parameters unconnected. All unconnected params use their default
value as the initial value. You can set the values of any unconnected params
the same way as any other variables by doing the following:

.. testcode:: parab

    top = Problem()
    root = top.root = Group()

    root.add('p', Paraboloid(), promotes=['x', 'y'])

    top.setup()

    # Set values for x and y
    top['x'] = 5.0
    top['y'] = 2.0

    top.run()

    print(top['p.f_xy'])

This can only be done after `setup` is called. Note that the promoted names
'x' and 'y' are used.

The new output should look like this:

.. testoutput:: parab
   :options: +ELLIPSIS

   47.0

Future tutorials will show more complex `Problems`.

.. _`paraboloid_optimization_tutorial`:

Optimization of the Paraboloid
------------------------------

Now that we have the paraboloid model set up, let's do a simple unconstrained
optimization. Let's find the minimum point on the Paraboloid over the
variables x and y. This requires the addition of just a few more lines.

First, we need to import the optimizer.

.. testcode:: parab

    from openmdao.api import ScipyOptimizer

The main optimizer built into OpenMDAO is a wrapper around Scipy's `minimize`
function. OpenMDAO supports 9 of the optimizers built into `minimize`. The
ones that will be most frequently used are SLSQP and COBYLA, since they are the
only two in the `minimize` package that support constraints. We will use
SLSQP because it supports OpenMDAO-supplied gradients.

.. testcode:: parab

        top = Problem()
        root = top.root = Group()

        # Initial value of x and y set in the IndepVarComp.
        root.add('p1', IndepVarComp('x', 13.0))
        root.add('p2', IndepVarComp('y', -14.0))
        root.add('p', Paraboloid())

        root.connect('p1.x', 'p.x')
        root.connect('p2.y', 'p.y')

        top.driver = ScipyOptimizer()
        top.driver.options['optimizer'] = 'SLSQP'

        top.driver.add_desvar('p1.x', lower=-50, upper=50)
        top.driver.add_desvar('p2.y', lower=-50, upper=50)
        top.driver.add_objective('p.f_xy')

        top.setup()

        # You can also specify initial values post-setup
        top['p1.x'] = 3.0
        top['p2.y'] = -4.0

        top.run()

        print('\n')
        print('Minimum of %f found at (%f, %f)' % (top['p.f_xy'], top['p.x'], top['p.y']))

Every driver has an `options` dictionary which contains important settings for the driver.
These settings tell `ScipyOptimizer` which optimization method to use, so here we
select 'SLSQP'. For all optimizers, you can specify a convergence tolerance
'tol' and a maximum number of iterations 'maxiter.'

Next, we select the parameters the optimizer will drive by calling
`add_param` and giving it the `IndepVarComp` unknowns that we have created. We
also set high and low bounds for this problem. It is not required to set
these (they will default to -1e99 and 1e99 respectively), but it is generally
a good idea.

Finally, we add the objective. You can use any `unknown` in your model as the
objective.

Once we have called setup on the model, we can specify the initial conditions
for the design variables just like we did with unconnected params.

Since SLSQP is a gradient based optimizer, OpenMDAO will call the `linearize` method
on the `Paraboloid` while calculating the total gradient of the objective
with respect to the two design variables. This is done automatically.

Finally, we made a change to the print statement so that we can print the
objective and the parameters. This time, we get the value by keying into the
problem instance ('top') with the full variable path to the quantities we
want to see. This is equivalent to what was shown in the first tutorial.

Putting this all together, when we run the model, we get output that looks
like this (note, the optimizer may print some things before this, depending on
settings):

.. testoutput:: parab
   :options: +ELLIPSIS

   ...
   Minimum of -27.333333 found at (6.666667, -7.333333)


Optimization of the Paraboloid with a Constraint
------------------------------------------------

Finally, let's take this optimization problem and add a constraint to it. Our
constraint takes the form of an inequality we want to satisfy: x - y >= 15.

First, we need to add one more import to the beginning of our model.

.. testcode:: parab

    from openmdao.api import ExecComp


We'll use an `ExecComp` to represent our constraint in the model. An ExecComp
is a shortcut that lets us easily create a component that defines a simple
expression for us.


.. testcode:: parab

    top = Problem()

    root = top.root = Group()

    root.add('p1', IndepVarComp('x', 3.0))
    root.add('p2', IndepVarComp('y', -4.0))
    root.add('p', Paraboloid())

    # Constraint Equation
    root.add('con', ExecComp('c = x-y'))

    root.connect('p1.x', 'p.x')
    root.connect('p2.y', 'p.y')
    root.connect('p.x', 'con.x')
    root.connect('p.y', 'con.y')

    top.driver = ScipyOptimizer()
    top.driver.options['optimizer'] = 'SLSQP'

    top.driver.add_desvar('p1.x', lower=-50, upper=50)
    top.driver.add_desvar('p2.y', lower=-50, upper=50)
    top.driver.add_objective('p.f_xy')
    top.driver.add_constraint('con.c', lower=15.0)

    top.setup()
    top.run()

    print('\n')
    print('Minimum of %f found at (%f, %f)' % (top['p.f_xy'], top['p.x'], top['p.y']))

Here, we added an ExecComp named 'con' to represent part of our
constraint inequality. Our constraint is "x - y >= 15", so we have created an
ExecComp that will evaluate the expression "x - y" and place that result into
the unknown 'con.c'. To complete the definition of the constraint, we also
need to connect our 'con' expression to 'x' and 'y' on the paraboloid.

Finally, we need to tell the driver to use the unknown "con.c" as a
constraint using the `add_constraint` method. This method takes the name of
the variable and an "upper" or "lower" bound. Here we give it a lower bound
of 15, which completes the inequality constraint "x - y >= 15".

OpenMDAO also supports the specification of double sided constraints, so if
you wanted to constrain x-y to lie on a band between 15 and 16 which is "16 > x-y > 15",
you would just do the following:

::

    top.driver.add_constraint('con.c', lower=15.0, upper=16.0)


So now, putting it all together, we can run the model and get this:

.. testoutput:: parab
   :options: +ELLIPSIS

   ...
   Minimum of -27.083333 found at (7.166667, -7.833333)

A new optimum is found because the original one was infeasible (i.e., that
design point violated the constraint equation).

.. tags:: Tutorials, Component, Paraboloid, Optimization
