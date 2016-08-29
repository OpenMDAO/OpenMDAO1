""" Useful numerical constants.

Attributes
----------
inf_bound : float
    This parameter is intended to be used to denote a infinite bound on
    a design variable or constraint.  The default value of 2.0e20 is
    large enough that it will trigger special treatment of the bound
    as infinite by optimizers like SNOPT and IPOPT, but is not so large
    as to cause overflow/underflow errors as numpy.inf sometimes can.
"""

inf_bound = 2.0e20
