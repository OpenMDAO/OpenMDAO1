.. _OpenMDAO-Profiling:

=========
Profiling
=========

This tutorial describes how to use OpenMDAO's simple instance based profiling
capability.  Python has several good profilers available for general python
code, but the advantage of OpenMDAO's simple profiler is that it limits the
functions of interest down to a relatively small number of important OpenMDAO
functions, which helps prevent information overload.  Also, the OpenMDAO
profiler lets you view the profiled functions grouped by the specific
problem, system, group, driver, or solver that called them.

To use profiling, you first have to call `setup_profiling`. This must happen
after your system tree has been defined. This is necessary because the setup
process traverses down the tree searching for instance methods to wrap with
profiling information, and if those instance methods don't exist yet then
they cannot be wrapped. You must pass a top level object, typically your
Problem object, to `setup_profiling`.

After profiling has been set up, you then call `activate_profiling` to
start collection of profiling data.  If for some reason you want to only
collect profiling data during a particular part of execution, you can call
`deactivate_profiling` to turn off collection.  For example:


.. testcode:: profile_activate

    from openmdao.api import setup_profiling, activate_profiling, deactivate_profiling

    prob = Problem()

    # define my model...

    setup_profiling(prob)
    activate_profiling()

    prob.setup()

    prob.run()

    deactivate_profiling()

    # do some other stuff that I don't want to profile...


There are a few advanced options to `setup_profiling`, but in general you
won't need them.  Consult the docstring to learn more.

After your script is finished running, you should have two new files,
`prof_raw.0` and `funcs_prof_raw.0` in your current directory.  If you happen
to have activated profiling for an MPI run, then you'll have a copy of those
two files for each MPI process, so `prof_raw.0`, `prof_raw.1`, etc.

There are two command scripts you can run on those raw data files.  The first
is `prof_totals`.  Running that on raw profiling files will give you CSV
formatted output containing total runtime and total number of calls for
each profiled function.  For example: `prof_totals prof_raw.*` might
give you output like the following:

::

    Function Name, Total Time, Calls
    .run, 2.38887190819, 1
    .solve_nonlinear, 2.38884210587, 1
    .Newton.solve, 2.38882899284, 1
    fc.solve_nonlinear, 1.8699491024, 1
    fc.Newton.solve, 1.86993718147, 1
    fc.solve_linear, 0.996086359024, 20
    fc.LinearGaussSeidel.solve, 0.994989395142, 20
    fc.subgroup.solve_linear, 0.947152614594, 20
    fc.subgroup.ScipyGMRES.solve, 0.945965051651, 20
    fc.subgroup.solve_nonlinear, 0.706930875778, 21
    fc.subgroup.RunOnce.solve, 0.706773519516, 21
    fc.subgroup.fs.solve_nonlinear, 0.705542802811, 21
    ...


The second command script is `viewprof`.  It generates an html file called
`profile_sunburst.html` that
uses a d3-based sunburst to show the function call tree. The file should
be viewable in any browser. Hovering over an arc in the sunburst will show the
function pathname, the local and total elapsed time for that function, and the
local and total number of calls for that function.  Clicking on an arc will
collapse the view so that that arc's function will become the center
circle and only functions called by that function will be visible.  The top
level center circle before any arc has been collapsed does not represent a
real function. Instead, it shows the sum of the elapsed times of all of the
top level functions as its local time, and the total time that profiling was
active as its total time.  If the total time is greater than the local time,
that indicates that some amount of time was taken up by functions that were
not being profiled.

The profiling data needed for the viewer is included directly in the html file,
so the file can be passed around and viewed by other people.  It does
however require network access in order to load the d3 library.

To pop up the viewer in a browser immediately, use the `--show` option, for
example:

::

    viewprof raw_prof.* --show


You should then see something like this:


.. figure:: profile_sunburst.png
   :align: center
   :alt: An example of a profile sunburst viewer

   An example of a profile sunburst viewer.
