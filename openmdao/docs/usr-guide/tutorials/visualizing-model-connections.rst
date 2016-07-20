.. _OpenMDAO-Visualizing-Model-Connections:

====================================================
Visualizing OpenMDAO Model Structure and Connections
====================================================

It can be difficult to understand the structure and connections in an OpenMDAO model. OpenMDAO provides a script to generate a diagram which visualizes both in an integrated view. The diagram makes use of a partition tree to display the structure and an N2 diagram to display the connections. In addition to helping the user understand an OpenMDAO model, this tool can also be used to detect bugs in OpenMDAO models.

Even though the diagram combines a partition tree and an N2 diagram, for simplicity, this document will refer to the combined diagram as an N2 diagram.

This tutorial will:
    - give background information on N2 diagrams
    - give an overview of an OpenMDAO N2 diagram
    - show you how to generate an N2 diagram
    - show a working N2 diagram and explain how to interact with it and understand what it is showing you

Background information on N2 diagrams
=====================================

An `N2 diagram <https://en.wikipedia.org/wiki/N2_chart>`_ , also referred to as an N 2 chart, N-squared diagram, or N-squared chart, is a diagram in the shape of a matrix, representing functional or physical interfaces between system elements. N2 diagrams have been used extensively to develop data interfaces.
A basic N2 diagram is shown in the figure below. 

 .. figure:: n2_chart_definition.jpg
   :align: center
   :alt: N2 Diagram Definition.

   N2 Diagram Definition taken from `NASA Systems Engineering Handbook. <http://web.stanford.edu/class/cee243/NASASE.pdf>`_

The system functions are placed on the diagonal; the remainder of the squares in the N x N matrix represent the
interface inputs and outputs. Where a blank appears, there is no interface between the respective functions. Data flows
in a clockwise direction between functions (e.g., the symbol F1 F2 indicates data flowing from function F1, to function
F2). The data being transmitted can be defined in the appropriate squares. The clockwise flow of data between functions
that have a feedback loop can be illustrated by a larger circle called a control loop.

Overview of an OpenMDAO N2 diagram
==================================

In OpenMDAO N2 diagrams, each cell on the diagonal represents Groups, Subsystems, and Outputs.

The partition tree is shown on the left, and the N^2 diagram is on the right.

The partition tree shows the structure of the OpenMDAO model starting with the root node on the left and its children to the right.  These children (from left to right) include Groups, Subsystems, and Outputs.  Each node's height in the partition tree are sized by the total number of leaf nodes; the more leaf nodes, the taller the node. 

When the partition tree is first loaded, all leaf nodes (the right most nodes) are Outputs.  Every node in the partition tree is in execution order from top to bottom.

The partition tree on the side corresponds to the groups/components in the model


Generating N2 diagrams of OpenMDAO models
=========================================

Generating the N2 diagram is very simple. The user only needs to call the `view_tree` function on a `Problem` object. The `Problem` object must call `setup()` before being passed to `view_tree`. There is no need to call `run` on the `Problem`.

::

    from openmdao.api import Problem

    from examples.beam_tutorial import BeamTutorial
    from openmdao.api import view_tree

    top = Problem()
    top.root = BeamTutorial()

    top.setup(check=False)
    view_tree(top, show_browser=False)


Here are the arguments for the view_tree function:


.. function:: def view_tree(system, viewer='collapse_tree', expand_level=9999, outfile='tree.html', show_browser=True)


   Generates a self-contained html file containing a tree viewer of the specified type.  Optionally pops up a web browser to view the file.

   :param problem: the Problem (after problem.setup()) for the desired tree.
   :param outfile: name of the output html file.  Defaults to 'partition_tree_n2.html'
   :param show_browser: if True, pop up a browser to view the generated html file. Defaults to True
   :type problem: Problem
   :type outfile: string
   :type show_browser: bool


Working Example of an N2 diagram
================================

Below is an example of the model contained in the example file `sellar_state_MDF_optimize.py`.

Here are some instructions on how to use it.

Partition Tree
--------------

Left clicking on a node in the partition tree will allow you to navigate to that node. Right clicking on a node will collapse/uncollapse it.

N2 Diagram
----------

To the right of the Partition Tree is the N^2 (N-Squared) Diagram.  The right most nodes of the partition tree are on the diagonal of the N^2 diagram.  The connections are listed on the off-diagonal of the N^2 diagram.  Connections go from source to target in a clockwise order, so a connection in the upper right goes in normal execution order, but a connection in the bottom left is a feedback.

Hovering over an on-diagonal element will show source-to-connection arrows going to and/or from that element.  Hovering over an off-diagonal upper right connection element will show the clockwise source-to-connection arrow.  Hovering over an off-diagonal lower left connection element will show the clockwise source-to-connection feedback arrow, along with the associated execution cycle going back to the source.  A click on any element in the N^2 diagram will allow those arrows to persist.


------------


.. raw:: html
   :file: n2_sellar_state.html






