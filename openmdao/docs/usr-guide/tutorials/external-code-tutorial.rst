External Code Tutorial
----------------------

This tutorial will show how to make use of the `ExternalCode` `Component` in OpenMDAO. MDAO problems 
frequently make use of external programs. These could be custom codes or commercial codes, for example.
They are often analysis codes for a specific discipline. If these codes do not have Python APIs, it is necessary to "wrap"
the codes in an OpenMDAO component that can run the code from the command line. OpenMDAO 
provides a `Component`, `ExternalCode`, that greatly simplifies doing that.

In this tutorial we will give an example based on a common scenario of a code that takes
its inputs from an 
input file, performs some computations, and then writes the results to an output file. 
`ExternalCode` supports multiple 
input and output files but for simplicity, this example only uses one of each.

This tutorial is based on the :ref:`Paraboloid Tutorial <paraboloid_tutorial>`, except in this case, 
we will be using an external code to do the computations.

For the tutorial, the external code will be a Python script that evaluates the paraboloid 
equation used in the paraboloid tutorial. Normally, since the external code is a Python script, 
it would make sense to turn this code into an OpenMDAO `Component`. But for the purposes of this 
tutorial, we need to make sure the external code can run on any user's machine and the one 
thing we are sure of is that the user, regardless of platform, has Python. Here is the code 
for this external code. It simply reads its inputs, `x` and `y`, from an external file, 
does the same computation as the `Paraboloid` `Component` in the :ref:`Paraboloid Tutorial <paraboloid_tutorial>`
and writes the output, `f_xy`, to an output file.

::

    import sys

    def paraboloid(input_filename, output_filename):
        with open(input_filename, 'r') as input_file:
            file_contents = input_file.readlines()
        x, y = [ float(f) for f in file_contents ]

        f_xy = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0

        with open( output_filename, 'w') as out:
            out.write('%f\n' % f_xy )

    if __name__ == "__main__":

        input_filename = sys.argv[1]
        output_filename = sys.argv[2]
        
        paraboloid(input_filename, output_filename)


Next we need to write the code that makes use of this external code.

::

    from __future__ import print_function

    from openmdao.core import Problem, Group
    from openmdao.components import ExternalCode, ParamComp

    class ParaboloidExternalCode(ExternalCode):
        def __init__(self):
            super(ParaboloidExternalCode, self).__init__()

            self.add_param('x', val=0.0)
            self.add_param('y', val=0.0)

            self.add_output('f_xy', val=0.0)

        def solve_nonlinear(self, params, unknowns, resids):
            """f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3
            """

            x = params['x']
            y = params['y']

            input_filepath = 'paraboloid_input.dat'
            output_filepath = 'paraboloid_output.dat'

            # Generate the input file for the paraboloid external code
            with open(input_filepath, 'w') as input_file:
                input_file.write('%f\n%f\n' % (x,y))

            # Run the paraboloid external code
            self.options['command'].extend([input_filepath, output_filepath])
            self.options['external_input_files'] = [input_filepath,]
            self.options['external_output_files'] = [output_filepath,]
            super(ParaboloidExternalCode, self).solve_nonlinear(params, unknowns, resids)

            # Parse the output file from the external code and set the value of f_xy
            with open( output_filepath, 'r') as output_file:
                f_xy = float( output_file.read() )

            unknowns['f_xy'] = f_xy


    top = Problem()
    top.root = root = Group()

    extcode = ParaboloidExternalCode()
    root.add('p', extcode)

    # Create and connect inputs
    root.add('p1', ParamComp('x', 3.0))
    root.add('p2', ParamComp('y', -4.0))
    root.connect('p1.x', 'p.x')
    root.connect('p2.y', 'p.y')

    # ExternalCode needs to know the actual command used to run it
    extcode.options['command'] = ['python', 'paraboloid_external_code.py',]

    # Run the ExternalCode Component
    top.setup()
    top.run()

    # Print the output
    print(root.p.unknowns['f_xy'])

Next we will go through each section and explain how this code works.

Building the ExternalCode Component
===================================

We need to import some OpenMDAO classes. We also import the `print_function` to
ensure compatibility between Python 2.x and 3.x. You don't need the import if
you are running in Python 3.x.

::

    from __future__ import print_function

    from openmdao.components.param_comp import ParamComp
    from openmdao.core.component import Component
    from openmdao.core.problem import Problem, Group


OpenMDAO provides a base class, `ExternalCode`, which you should inherit from to build
your wrappers for external codes. 

::

    class ParaboloidExternalCode(ExternalCode):

This code defines the input parameters of the `ParaboloidExternalCode`, `x` and `y`, and
initializes them to *0.0*. These will be design variables which could be used to
minimize the output when doing optimization but in this example will only be used
for analysis. It also defines the explicit output, `f_xy`.

::

        def __init__(self):
            super(ParaboloidExternalCode, self).__init__()

            self.add_param('x', val=0.0)
            self.add_param('y', val=0.0)

            self.add_output('f_xy', val=0.0)

The `solve_nonlinear` method is responsible for calculating outputs for a
given set of parameters. The parameters are given in the `params` dictionary
that is passed in to this method. Similarly, the outputs are assigned values
using the `unknowns` dictionary that is passed in.

Since we are making use of an external code to do the computation, we have to feed it
the input parameter values via an input file and then also tell it the name of the 
output file to write the resulting output unknown to.  

::

        def solve_nonlinear(self, params, unknowns, resids):
            """f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3
            """

            # Just hardcode the filenames for this tutorial
            input_filepath = 'paraboloid_input.dat'
            output_filepath = 'paraboloid_output.dat'

            # Generate the input file for the paraboloid external code
            x = params['x']
            y = params['y']
            with open(input_filepath, 'w') as input_file:
                input_file.write('%f\n%f\n' % (x,y))

            # Run the paraboloid external code
            self.options['command'].extend([input_filepath, output_filepath])
            self.options['external_input_files'] = [input_filepath,]
            self.options['external_output_files'] = [output_filepath,]
            super(ParaboloidExternalCode, self).solve_nonlinear(params, unknowns, resids)

            # Parse the output file from the external code and set the value of f_xy
            with open( output_filepath, 'r') as output_file:
                f_xy = float( output_file.read() )
            unknowns['f_xy'] = f_xy

The definition of the `ParaboloidExternalCode` `Component` class is now complete. We will now
make use of this class to run a model.

Setting up and running the model
================================

You will notice that this code to run the model is very similar to the code used 
for the :ref:`Paraboloid Tutorial <paraboloid_tutorial>`. In fact, the only 
difference is that instead of creating a `Paraboloid` `Component`, we 
create a `ParaboloidExternalCode` `Component`.

An instance of an OpenMDAO `Problem` is always the top object for running a
model. Each `Problem` in OpenMDAO must contain a root `Group`. A `Group` is a
`System` that contains other `Components` or `Groups`.

This code instantiates a `Problem` object and sets the root to be an empty `Group`.

::

    if __name__ == "__main__":

        top = Problem()
        root = top.root = Group()

Now it is time to add components to the empty group. `ParamComp`
is a `Component` that provides the source for a variable which we can later give
to a `Driver` as a design variable to control.

Then we add the paraboloid external code `Component`, giving it the name 'p'.

::

    extcode = ParaboloidExternalCode()
    root.add('p', extcode)

We created two `ParamComps` (one for each param on the `ParaboloidExternalCode`
component), gave them names, and added them to the root `Group`. The `add`
method takes a name as the first argument, and a `Component` instance as the
second argument.

::

    root.add('p1', ParamComp('x', 3.0))
    root.add('p2', ParamComp('y', -4.0))

Then we connect up the outputs of the `ParamComps` to the parameters of the
`ParaboloidExternalCode`. Notice the dotted naming convention used to refer to variables.
So, for example, `p1` represents the first `ParamComp` that we created to set
the value of `x` and so we connect that to parameter `x` of the `ParaboloidExternalCode`.
Since the `ParaboloidExternalCode` is named `p` and has a parameter
`x`, it is referred to as `p.x` in the call to the `connect` method.

::

    root.connect('p1.x', 'p.x')
    root.connect('p2.y', 'p.y')

Before we can run our model we need to do some setup. This is done using the
`setup` method on the `Problem`. This method performs all the setup of vector
storage, data transfer, etc.., necessary to perform calculations. Calling
`setup` is required before running the model.

::

    top.setup()

Now we can run the model using the `run` method of `Problem`.

::

    top.run()

Finally, we print the output of the `ParaboloidExternalCode` `Component` using the
dictionary-style method of accessing the outputs from a `Component` instance.

::

    print(root.p.unknowns['f_xy'])

