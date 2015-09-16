.. index:: Kriging MetaModel Tutorial

MetaModel Component
---------------------------

A number of utility components come packaged with OpenMDAO. `MetaModel`
is one that lets you quickly create a component with surrogate models
used to compute the outputs based on training data. You can set up
a `MetaModel` instance with as many parameters and outputs as you like,
and you can also use a different surrogate model for each output.

.. note::

    What's the difference between a `MetaModel` and a surrogate model? In
    OpenMDAO, "surrogate model" refers to the model for a single response, and
    `MetaModel` represents a collection of surrogate models trained at the
    same locations in the design space.

This code will set up a really simple `Group` with only a single
`MetaModel` instance, using one parameter and two outputs.

.. testcode:: krig_example

    from __future__ import print_function

    import sys
    import numpy as np

    from openmdao.core import Group, Component
    from openmdao.components import MetaModel
    from openmdao.surrogate_models import KrigingSurrogate, FloatKrigingSurrogate

    class TrigMM(Group):
        ''' FloatKriging gives responses as floats '''

        def __init__(self):
            super(TrigMM, self).__init__()

            # Create meta_model for f_x as the response
            sin_mm = self.add("sin_mm", MetaModel())
            sin_mm.add_desvar('x', val=0.)
            sin_mm.add_output('f_x:float', val=0., surrogate=FloatKrigingSurrogate())
            sin_mm.add_output('f_x:norm_dist', val=(0.,0.), surrogate=KrigingSurrogate())

Now we'll setup the problem and set some training data. Here
we just generate the data on the fly, but normally you would have
pre-generated this data and then would just import it and use it.

.. testcode:: krig_example

    from openmdao.core import Problem

    prob = Problem()
    prob.root = TrigMM()
    prob.setup()

    #traning data is just set manually. No connected input needed, since
    #  we're assuming the data is pre-existing
    prob['sin_mm.train:x'] = np.linspace(0,10,20)
    prob['sin_mm.train:f_x:float'] = np.sin(prob['sin_mm.train:x'])
    prob['sin_mm.train:f_x:norm_dist'] = np.cos(prob['sin_mm.train:x'])

    prob['sin_mm.x'] = 2.1 #prediction happens at this value
    prob.run()
    print('float predicted:', '%3.4f'%prob['sin_mm.f_x:float']) #predicted value
    print('float actual: ', '%3.4f'%np.sin(2.1))
    print('norm_dist predicted:', '%3.4f,'%prob['sin_mm.f_x:norm_dist'][0], '%3.4e'%prob['sin_mm.f_x:norm_dist'][1]) #predicted value
    print('norm_dist actual: ', '%3.4f'%np.cos(2.1))


You should get some output that looks like this:

.. testoutput:: krig_example

   float predicted: 0.8632
   float actual:  0.8632
   norm_dist predicted: -0.5048, ... 
   norm_dist actual:  -0.5048

Notice that one of the outputs is non-float data. Some surrogate models
(like Kriging), can return non-float data like integers, strings, or
probability distributions.
