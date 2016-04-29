""" VecWrapper 'wrapper' that is used for component-wise complex step."""

from six import iteritems, iterkeys

import numpy as np

# Don't autodoc anything
__all__ = []


class ComplexStepTgtVecWrapper(object):
    """ Acts like a TgtVecWrapper to the user, but allows a complex value to
    be returned from the stepped variable."""

    def __init__(self, vec):

        self.vecwrap = vec
        self.vec = vec.vec
        self.step_var = None
        self.step_val = None

    def __getitem__(self, name):
        """
        Retrieve unflattened value of named var.

        Args
        ----
        name : str
            Name of variable to get the value for.

        Returns
        -------
            The unflattened value of the named variable.
        """
        if name == self.step_var:
            return self.step_val.reshape(self.vecwrap[name].shape)

        return self.vecwrap._dat[name].get()

    def __len__(self):
        """
        Returns
        -------
            The number of keys (variables) in this vector.
        """
        return len(self.vecwrap)

    def __contains__(self, key):
        """
        Returns
        -------
            A boolean indicating if the given key (variable name) is in this vector.
        """

        return key in self.vecwrap

    def __iter__(self):
        """
        Returns
        -------
            A dictionary iterator over the items in _dat.
        """
        return iter(self.vecwrap)

    def keys(self):
        """
        Returns
        -------
        list or KeyView (python 3)
            the keys (variable names) in this vector.
        """
        return self.vecwrap.keys()

    def iterkeys(self):
        """
        Returns
        -------
        iter of str
            the keys (variable names) in this vector.
        """
        return self.vecwrap.iterkeys()

    def metadata(self, name):
        """
        Returns the metadata for the named variable.

        Args
        ----
        name : str
            Name of variable to get the metadata for.

        Returns
        -------
        dict
            The metadata dict for the named variable.

        Raises
        -------
        KeyError
            If the named variable is not in this vector.
        """
        return self.vecwrap._dat[name].meta

    def set_complex_var(self, name):
        """
        Specifies the current input variable that will be complex stepped.

        Args
        ----
        name : str
            Name of variable to get the metadata for.
        """

        if name is None:
            self.step_var = None
            self.step_val = None
            return

        var = self.vecwrap._dat[name].val
        self.step_var = name
        self.step_val = np.zeros(len(var), dtype=np.complex)
        self.step_val[:] = var

    def step_complex(self, idx, stepsize):
        """
        Specifies the current input variable that will be complex stepped.

        Args
        ----
        idx : integer
            Index into step_var flat vector to apply the step.

        stepsize : float
            Step value. Omit the j.
        """
        self.step_val[idx] += 1j*stepsize


class ComplexStepSrcVecWrapper(object):
    """ Acts like a SrcVecWrapper to the user, but allows a complex value to
    be retrieved from and set into any variable. """

    def __init__(self, vec):

        self.vecwrap = vec
        self.vec = vec.vec
        self.step_var = None
        self.step_val = None
        self.vals = {}

        # Make complex copies of every unknown or state
        for name, val in iteritems(vec):
            self.vals[name] = np.zeros(val['shape'], dtype=np.complex)
            self.vals[name][:] = vec[name]

    def __getitem__(self, name):
        """
        Retrieve unflattened value of named var.

        Args
        ----
        name : str
            Name of variable to get the value for.

        Returns
        -------
            The unflattened value of the named variable.
        """
        if name == self.step_var:
            return self.step_val.reshape(self.vecwrap[name].shape)

        return self.vals[name]

    def __setitem__(self, name, value):
        """
        Set the value of the named variable.

        Args
        ----
        name : str
            Name of variable to get the value for.

        value :
            The unflattened value of the named variable.
        """
        self.vals[name] = value

    def __len__(self):
        """
        Returns
        -------
            The number of keys (variables) in this vector.
        """
        return len(self.vecwrap)

    def __contains__(self, key):
        """
        Returns
        -------
            A boolean indicating if the given key (variable name) is in this vector.
        """

        return key in self.vecwrap

    def __iter__(self):
        """
        Returns
        -------
            A dictionary iterator over the items in _dat.
        """
        return iter(self.vecwrap)

    def keys(self):
        """
        Returns
        -------
        list or KeyView (python 3)
            the keys (variable names) in this vector.
        """
        return self.vecwrap.keys()

    def iterkeys(self):
        """
        Returns
        -------
        iter of str
            the keys (variable names) in this vector.
        """
        return self.vecwrap.iterkeys()

    def metadata(self, name):
        """
        Returns the metadata for the named variable.

        Args
        ----
        name : str
            Name of variable to get the metadata for.

        Returns
        -------
        dict
            The metadata dict for the named variable.

        Raises
        -------
        KeyError
            If the named variable is not in this vector.
        """
        return self.vecwrap._dat[name].meta

    def flat(self, name):
        """
        Returns flattened value of variable in name.

        Args
        ----
        name : str
            Name of variable to flatten.

        Returns
        -------
        ndarray
            Variable value.

        Raises
        -------
        KeyError
            If the named variable is not in this vector.
        """
        val = self.vals[name]
        if isinstance(val, np.ndarray):
            return val.flatten()
        else:
            return np.array([val])

    def set_complex_var(self, name):
        """
        Specifies the current input variable that will be complex stepped.

        Args
        ----
        name : str
            Name of variable to get the metadata for.
        """

        if name is None:
            self.step_var = None
            self.step_val = None
            return

        var = self.vecwrap._dat[name].val
        self.step_var = name
        self.step_val = np.zeros(len(var), dtype=np.complex)
        self.step_val[:] = var

    def step_complex(self, idx, stepsize):
        """
        Specifies the current input variable that will be complex stepped.

        Args
        ----
        idx : integer
            Index into step_var flat vector to apply the step.

        stepsize : float
            Step value. Omit the j.
        """
        self.step_val[idx] += 1j*stepsize

    def _scale_values(self):
        """ Applies the 'scaler' or 'resid_scaler' to the quantities sitting
        in the unknown or residual vectors.
        """
        wrap = self.vecwrap
        for name, acc in iteritems(wrap._dat):
            meta = acc.meta
            if 'scaler' in meta and wrap.vectype == 'u':
                scaler = 1.0/meta['scaler']
                self.vals[name] *= scaler
                acc.disable_scale = False
            elif 'resid_scaler' in meta and wrap.vectype == 'r':
                scaler = 1.0/meta['resid_scaler']
                self.vals[name] *= scaler
                acc.disable_scale = False

    def _disable_scaling(self):
        """ Turns off automatic scaling when getting a value via the
        dictionary accessor. It is only turned off in the unknowns vector
        during solve_nonlinear to allow the user to get a reference to the
        unknown so that it can be flled by index.
        """
        self.vecwrap._disable_scaling()
