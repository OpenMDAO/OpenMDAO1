.. _Design-Doc:

============
Design
============

relationships between problem, group, driver, component, etc.
""" System: Base class for systems in OpenMDAO."""

"""   Component: Base class for a Component system. The Component can declare
variables and operates on its params to produce unknowns, which can be
explicit outputs or implicit states.
"""

""" Base class for drivers in OpenMDAO. Drivers can only be placed in a
Problem, and every problem has a Driver. Driver is the simplest driver that
runs (solves using solve_nonlinear) a problem once.
"""

"""Group: A system that contains other systems."""

""" The Problem is always the top object for running an OpenMDAO
model.
"""
