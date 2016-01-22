File Passing Tutorial
======================

References to files can be passed between Components in OpenMDAO using variables
called `FileRefs`.  A `FileRef` is just an object that contains a file name and
an absolute file path which is calculated by the framework. The *fname* attribute
can be a simple name or a name that includes a relative or absolute
directory path.

Calculation of Absolute Directory Paths
---------------------------------------

During setup, OpenMDAO determines the absolute file system path for each
`FileRef` variable based on the directory path of the `Component`
that contains it.  The process works like this:

1) Starting at the root `System` of the tree, we calculate its absolute directory
   based on its 'directory' attribute.  If the 'directory' attribute is empty,
   then the absolute directory is just the current working directory. If
   *directory* contains a relative pathname, then the absolute directory is
   the current working directory plus the relative path.  If *directory* is
   already an absolute path, then we just use that.

2) For each child `System` in the tree, we calculate its absolute directory
   based on the absolute directory we've already calculated for its parent
   `System`, in the same manner as in step 1, except that instead of the
   current working directory, we use the parent absolute directory as our
   starting point.

3) In each `Component` we encounter as we traverse the tree, after we've
   calculated its absolute directory, we look for any `FileRef` variables
   it may have.  We set the absolute directory for each `FileRef` in a
   similar way as in step 2, but in this case the `Component` is the parent,
   so we use its absolute directory as our starting point in determining
   the absolute path of the `FileRef`.

.. note::

    Sometimes you may not want to hard-code the directory name. If you want
    to delay picking a name until runtime, you can specify directory as a
    function. If *directory* is a function, we will call that function,
    passing in the rank for the current process.  That function should return
    a string containing either a relative or absolute path, which we will
    resolve to an absolute directory as mentioned above.


Using FileRefs
--------------

So lets make some components that pass FileRefs between them.  We'll just use
ascii files here to keep things as simple as possible, but FileRefs can be
binary if you set *binary=True* in the metadata when you add them to a
component.

First, we'll make a simple component that takes a single parameter, does a
simple calculation, then writes the result to a file.


.. testsetup:: FileRef1, FileRef2

    import os, tempfile, shutil, errno
    startdir = os.getcwd()
    tmpdir = tempfile.mkdtemp()
    os.chdir(tmpdir)


.. testcleanup:: FileRef1, FileRef2

    os.chdir(startdir)
    try:
        shutil.rmtree(tmpdir)
    except OSError as e:
        # If directory already deleted, keep going
        if e.errno not in (errno.ENOENT, errno.EACCES, errno.EPERM):
            raise e

.. testcode:: FileRef1, FileRef2, FileRef3

    from openmdao.api import Problem, Group, Component, FileRef

    class FoutComp(Component):
        """A component that writes out a file containing a number."""

        def __init__(self):
            super(FoutComp, self).__init__()

            # add a simple parameter that we can use to calculate the
            # number we write to our output file
            self.add_param('x', 1.0)

            # add an output FileRef for our output file 'dat.out'
            self.add_output("outfile", FileRef("dat.out"))

        def solve_nonlinear(self, params, unknowns, resids):
            # do some simple calculation
            val = params['x'] * 2.0 + 1.0

            # write the new value to our output FileRef
            with unknowns['outfile'].open('w') as f:
                f.write(str(val))

Now we need a component to read a number from our first component's output
file and use that to calculate a new number.

.. testcode:: FileRef1, FileRef2

    class FinComp(Component):
        """A component that reads a file containing a number."""

        def __init__(self):
            super(FinComp, self).__init__()

            # add an input FileRef for our input file 'dat.in'
            self.add_param("infile", FileRef("dat.in"))

            # here's the output we'll calculate using the number we read
            # from our input FileRef
            self.add_output('y', 1.0)

        def solve_nonlinear(self, params, unknowns, resids):
            # read the number from our input FileRef
            with params['infile'].open('r') as f:
                val = float(f.read())

            # now calculate our new output value
            unknowns['y'] = val + 7.0

Now we have our two file transferring components, so we can build our model.

.. testcode:: FileRef1

    p = Problem(root=Group())
    outfilecomp = p.root.add("outfilecomp", FoutComp())
    infilecomp = p.root.add("infilecomp", FinComp())

    # connect our two FileRefs together
    p.root.connect("outfilecomp.outfile", "infilecomp.infile")

    p.setup()


We'll set a value of 3.0 in our first component's *x* value.  That should
give us a *y* value in our second component of 14.0.

.. testcode:: FileRef1

    p['outfilecomp.x'] = 3.0

    p.run()

    print(p['infilecomp.y'])


.. testoutput:: FileRef1

    14.0

In this example, our files were both in the same directory, but you can control
where they are found by modifying the *directory* attribute of systems in the
tree.  For example, if we wanted *outfilecomp.outfile* to be located in a *sub1*
subdirectory, we could do the following:

.. testcode:: FileRef2

    p = Problem(root=Group())
    outfilecomp = p.root.add("outfilecomp", FoutComp())

    # specify the subdirectory here
    outfilecomp.directory = 'sub1'

    # since 'sub1' doesn't exist, we need to tell the component to create it.
    # otherwise, we'll get an error that the directory doesn't exist.
    outfilecomp.create_dirs = True

    infilecomp = p.root.add("infilecomp", FinComp())

    # connect our two FileRefs together
    p.root.connect("outfilecomp.outfile", "infilecomp.infile")

    p.setup()


Notice that none of the code in our components or any of our other configuration
code has changed.  When we run this problem, we get the same
answer as before.

.. testcode:: FileRef2

    p['outfilecomp.x'] = 3.0

    p.run()

    print(p['infilecomp.y'])


.. testoutput:: FileRef2

    14.0


FileRefs under MPI
------------------

When running under MPI, there are certain situations where you may need to
create subdirectories dynamically based on the rank of the current MPI process.
You can accomplish that by assigning a function to a system's directory instead
of just a simple string.  For example, suppose we had a group in our model
that we wanted to perform parallel finite difference on, and that group happened
to have output `FileRefs` in it.  In that situation, different MPI processes
would try to write to the same output file at the same time.  In order to
prevent this, we can specify that in each MPI process, our group will have a
directory specific to that process.  Assigning *directory* to a function
instead of a string will let us do that.  For example, let's say we want our
group to write its files in a subdirectory called 'foo_n', where 'n' is the
rank of the current process.  In that case, setting our group's *directory*
would look like this:

::

    mygrp.directory = lambda rank: "foo_%d" % rank
    mygrp.create_dirs = True  # create the directories if they don't exist

The function you assign to *directory* should expect a single argument that is
the rank of the current process, and it should return the desired directory string.
Note that it's also valid to assign a method of your component to *directory* if
you happen to need more information than just the rank in order to
determine the directory name.  For example:

.. testcode:: FileRef3

    class MyComp(FoutComp):
        def get_dirname(self, rank):
            return "%s_%d" % (self.name, rank)

    mycomp = MyComp()
    mycomp.directory = mycomp.get_dirname
    mycomp.create_dirs = True
