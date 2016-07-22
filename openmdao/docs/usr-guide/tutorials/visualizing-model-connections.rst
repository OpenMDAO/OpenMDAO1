.. _OpenMDAO-Visualizing-Model-Connections:

====================================================
Visualizing OpenMDAO Model Structure and Connections
====================================================

It can be difficult to understand the structure and connections in an OpenMDAO model. OpenMDAO provides a script to generate a diagram which visualizes both in an integrated view. The diagram makes use of a partition tree to display the structure and an N2 diagram to display the connections. In addition to helping the user understand an OpenMDAO model, this tool can also be used to detect bugs in OpenMDAO models.

This tutorial will:
    - give background information on N2 diagrams
    - show a working partition tree/N2 diagram and explain how to interact with it and understand what it is showing you
    - show you how to generate an partition tree/N2 diagram

Background information on N2 diagrams
=====================================

An `N2 diagram <https://en.wikipedia.org/wiki/N2_chart>`_ , also referred to as an N 2 chart, N-squared diagram, or N-squared chart, is a diagram in the shape of a matrix, representing functional or physical interfaces between system elements. N2 diagrams have been used extensively to develop data interfaces. A basic N2 diagram is shown in the figure below. 

 .. figure:: n2_chart_definition.jpg
   :align: center
   :alt: N2 Diagram Definition.

   N2 Diagram Definition taken from `NASA Systems Engineering Handbook. <http://web.stanford.edu/class/cee243/NASASE.pdf>`_

The system functions are placed on the diagonal; the remainder of the squares in the N x N matrix represent the interface inputs and outputs. Where a blank appears, there is no interface between the respective functions. Data flows in a clockwise direction between functions (e.g., the symbol F1 F2 indicates data flowing from function F1, to function F2). The data being transmitted can be defined in the appropriate squares. The clockwise flow of data between functions that have a feedback loop can be illustrated by a larger circle called a control loop.

Working Example of an OpenMDAO Partition Tree/N2 Diagram
========================================================

This section will give an overview of an OpenMDAO partition tree/N2 Diagram and how to use it. The working example below shows the OpenMDAO model contained in the example file `sellar_state_MDF_optimize.py`.

In an OpenMDAO partition tree/N2 diagram, the partition tree is on the left, and the N2 diagram is on the right.

Partition Tree
--------------

The partition tree shows the structure of the OpenMDAO model starting with the root node on the left and its children to the right.  These children (from left to right) include Groups, Subsystems, and Outputs.  Each node's height in the partition tree are sized by the total number of leaf nodes; the more leaf nodes, the taller the node. 

When the partition tree is first loaded, all leaf nodes (the right most nodes) are Outputs.  Every node in the partition tree is in execution order from top to bottom.

The Legend below the diagram explains the colors of the nodes in the partition tree.

Left clicking on a node in the partition tree will allow you to navigate to that node. Right clicking on a node will collapse/uncollapse it.

You can also control what is displayed in the partition tree using the buttons in the Collapse Algorithms and Navigation sections above the diagram.

N2 Diagram
----------

To the right of the partition tree is the N2 Diagram.  The right most nodes of the partition tree are on the diagonal of the N2 diagram.  The connections are listed on the off-diagonal of the N2 diagram.  Connections go from source to target in a clockwise order, so a connection in the upper right goes in normal execution order, but a connection in the bottom left is a feedback.

Hovering over an on-diagonal element will show source-to-connection arrows going to and/or from that element.  Hovering over an off-diagonal upper right connection element will show the clockwise source-to-connection arrow.  Hovering over an off-diagonal lower left connection element will show the clockwise source-to-connection feedback arrow, along with the associated execution cycle going back to the source.  A click on any element in the N2 diagram will allow those arrows to persist. You can clear the arrows using the Clear Arrows button above the diagram.

Here are some examples of what you can learn from the example diagram:

    - if you hover over on the diagonal element for y2, the arrows show that y2 depends on y1 and z1. It also shows that y2_command, con2, and obj depend on y2
    - if you hover over the diagonal element for y2_command, it shows that y2_command depends on y2 and also there is a feedback dependency where y1 depends on y2_command
    - the most interesting display occurs when you hover over the element below the diagonal. It shows the connection that make up a cycle in the model. 

The legend below the diagram explains the symbols used in the diagram.

------------


.. raw:: html
   :file: n2_sellar_state.html



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


.. function:: def view_tree(problem, outfile='partition_tree_n2.html', show_browser=True, offline=True, embed=False)


   Generates a self-contained html file containing a tree viewer of the specified type.  Optionally pops up a web browser to view the file.

   :param problem: the Problem (after problem.setup()) for the desired tree.
   :param outfile: name of the output HTML file.  Defaults to 'partition_tree_n2.html'
   :param show_browser: if True, pop up a browser to view the generated HTML file. Defaults to True
   :param offline: if True, embed the JavaScript d3 library into the generated HTML file so that the tree can be viewed
       offline without an internet connection.  Otherwise if False, have the HTML request the latest d3 file
       from https://d3js.org/d3.v4.min.js when opening the HTML file.
       Defaults to True
   :param embed: if True, export only the innerHTML that is between the body tags, used for embedding the viewer into another HTML file.
       If False, create a standalone HTML file that has the DOCTYPE, html, head, meta, and body tags.
       Defaults to False
   :type problem: Problem
   :type outfile: string
   :type show_browser: bool
   :type offline: bool
   :type embed: bool


