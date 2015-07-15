
.. _Conversion-Guide:

_____________________________
Introduction
_____________________________

========================
Purpose of This Document
========================

The purpose behind the OpenMDAO Conversion Guide is to help users of previous
versions of OpenMDAO (versions up to and including 0.13.0) to change their models
over to the new OpenMDAO 1.0.0 design.  This will require some re-thinking and
re-structuring.  If you are new to OpenMDAO, you should be able to start writing
new models, and this guide should not pertain to you.  You should check out our
OpenMDAO User Guide_.

.. _Guide: ../usr-guide/index.html

If you do not have OpenMDAO v 1.0 installed, you should first view our Getting
Started Guide_.  Then we would recommend becoming familiar with the new building
blocks of OpenMDAO in the User Guide's 'Basics_' section

.. _Design: ../getting-started/basics.html


Conceptually, the core building blocks of OpenMDAO 1.0 are similar to those
found in previous versions, but the syntax you use to define those building blocks
is quite different.  This guide will start by describing the differences you'll
see when defining a Component.  Then we'll move on to the process of connecting
your Components and building your model.

========================
Declaring your Component
========================

We'll start off by defining a very simple component, one that has a single
input ``x`` and a single output ``y``, both having a value of type ``float``.
When the component runs, it will assign the value of `x * 2` to `y`.

In old OpenMDAO, our component class would look like this:

::
    from openmdao.main.api import Component
    from openmdao.main.datatypes.api import Float

    class Times2(Component):
        # variables are declared at the class level
        x = Float(1.0, iotype='in', desc='my var x')
        y = Float(2.0, iotype='out', desc='my var y')

        def execute(self):
            self.y = self.x * 2.0


The new way of defining the same component is:

::
    from openmdao.core.component import Component

    class Times2(Component):
        def __init__(self):
            # variables are added in the __init__ method
            self.add_param('x', 1.0, desc='my var x')
            self.add_output('y', 2.0, desc='my var y')

        def solve_nonlinear(self, params, unknowns, resids):
            unknowns['y'] = params['x'] * 2.0


=========================
Changes in function names
=========================

[Clippy vs Classic]


=================
Concrete Examples
=================

Paraboloid (link)

Sellar Problem (link)

=======
Support
=======

Moving your previous models to OpenMDAO 1.0 may be an arduous process, but it
will be one that we feel will be worth the effort.  If things get confusing or
difficult, we're happy to help.  [Link to forums?  Link to the openmdao tag on
Stack Overflow?  support@openmdao.org email address?]
