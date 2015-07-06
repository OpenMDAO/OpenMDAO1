.. _Design-Doc:

============
Design
============

Overview
--------

[First we need a high-level discussion of the relationships between Problem,
System, Group, Driver, Component, etc.  This should NOT mention how things used
to be.  This doc is only concerned with the present.]

Component
---------

Base class for a Component system. The Component can declare
variables and operates on its params to produce unknowns, which can be
explicit outputs or implicit states.

System
------

Base class for systems in OpenMDAO.



Driver
------

Driver is the base class for drivers in OpenMDAO, it is the simplest driver possible,
running a problem once. Drivers can only be placed in a
Problem, and every problem has a Driver.  By default, a driver solves using solve_nonlinear.


Group
------

A system that contains other systems.


Problem
-------

Is always the top object for running an OpenMDAO model.


[perhaps we could make a few diagrams to show relationships?]
