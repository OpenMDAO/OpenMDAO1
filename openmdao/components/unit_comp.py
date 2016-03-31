""" Class definition for UnitComp, a component that explicitly converts units."""

from openmdao.core.component import Component


class UnitComp(Component):
    """
    A Component that converts the input into the requested units.

    Args
    ----
    shape : tuple or int
        A tuple (or int if one-dimensional) that describes the shape of the
        input and output.
    param_name : str
        A string containing the name for the input parameter.
    out_name : str
        A string containing the name for the output variable.
    units : str
        A string containing the units to which inputs are converted.

    Options
    -------
    fd_options['force_fd'] :  bool(False)
        Set to True to finite difference this system.
    fd_options['form'] :  str('forward')
        Finite difference mode. (forward, backward, central) You can also set to 'complex_step' to peform the complex step method if your components support it.
    fd_options['step_size'] :  float(1e-06)
        Default finite difference stepsize
    fd_options['step_type'] :  str('absolute')
        Set to absolute, relative
    fd_options['extra_check_partials_form'] :  None or str
        Finite difference mode: ("forward", "backward", "central", "complex_step")
        During check_partial_derivatives, you can optionally do a
        second finite difference with a different mode.
    fd_options['linearize'] : bool(False)
        Set to True if you want linearize to be called even though you are using FD.

    """

    def __init__(self, shape, param_name, out_name, units):
        super(UnitComp, self).__init__()

        self.param_name = param_name
        self.out_name = out_name
        self.shape = shape

        if param_name == out_name:
            msg = "UnitComp param_name cannot match out_name: '{name}'"
            raise ValueError(msg.format(name=param_name))

        self.add_param(param_name, shape=shape, units=units)
        self.add_output(out_name, shape=shape, units=units)

    def solve_nonlinear(self, params, unknowns, resids):
        """For `UnitComp`, just pass on the incoming values.

        Args
        ----
        params : `VecWrapper`
            `VecWrapper` containing parameters. (p)

        unknowns : `VecWrapper`
            `VecWrapper` containing outputs and states. (u)

        resids : `VecWrapper`, optional
            `VecWrapper` containing residuals. (r)
        """
        unknowns[self.out_name] = params[self.param_name]

    def apply_linear(self, params, unknowns, dparams, dunknowns, dresids,
                     mode):
        """
        Multiplies incoming vector by the Jacobian (fwd mode) or the
        transpose Jacobian (rev mode).

        Args
        ----
        params : `VecWrapper`
            `VecWrapper` containing parameters. (p)

        unknowns : `VecWrapper`
            `VecWrapper` containing outputs and states. (u)

        dparams : `VecWrapper`
            `VecWrapper` containing either the incoming vector in forward mode
            or the outgoing result in reverse mode. (dp)

        dunknowns : `VecWrapper`
            In forward mode, this `VecWrapper` contains the incoming vector for
            the states. In reverse mode, it contains the outgoing vector for
            the states. (du)

        dresids : `VecWrapper`
            `VecWrapper` containing either the outgoing result in forward mode
            or the incoming vector in reverse mode. (dr)

        mode : string
            Derivative mode, can be 'fwd' or 'rev'.
        """

        if mode == 'fwd':
            dresids[self.out_name] += dparams[self.param_name]

        elif mode == 'rev':
            dparams[self.param_name] += dresids[self.out_name]
