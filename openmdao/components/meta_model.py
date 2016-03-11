""" Metamodel provides basic Meta Modeling capability."""

import sys
import numpy as np
from copy import deepcopy

from openmdao.core.component import Component, _NotSet
from six import iteritems


class MetaModel(Component):
    """Class that creates a reduced order model for outputs from
    parameters. Each output may have it's own surrogate model.
    Training inputs and outputs are automatically created with
    'train:' prepended to the corresponding parameter/output name.

    For a Float variable, the training data is an array of length m.

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

    def __init__(self):
        super(MetaModel, self).__init__()

        # This surrogate will be used for all outputs that don't have
        # a specific surrogate assigned to them
        self.default_surrogate = None

        # keep list of params and outputs that are not the training vars
        self._surrogate_param_names = []
        self._surrogate_output_names = []

        # training will occur on first execution
        self.train = True
        self._training_input = np.zeros(0)
        self._training_output = {}

        # When set to False (default), the metamodel retrains with the new
        # dataset whenever the training data values are changed. When set to
        # True, the new data is appended to the old data and all of the data
        # is used to train.
        self.warm_restart = False

        # keeps track of which sur_<name> slots are full
        self._surrogate_overrides = set()

        self._input_size = 0

    def add_param(self, name, val=_NotSet, training_data=None, **kwargs):
        """ Add a `param` input to this component and a corresponding
        training parameter.

        Args
        ----
        name : string
            Name of the input.

        val : float or ndarray or object
            Initial value for the input.

        training_data : float or ndarray
            training data for this variable. Optional, can be set
            by the problem later.
        """
        if training_data is None:
            training_data = []

        super(MetaModel, self).add_param(name, val, **kwargs)
        super(MetaModel, self).add_param('train:'+name, val=training_data, pass_by_obj=True)

        input_size = self._init_params_dict[name]['size']

        self._surrogate_param_names.append((name, input_size))
        self._input_size += input_size

    def add_output(self, name, val=_NotSet, training_data=None, **kwargs):
        """ Add an output to this component and a corresponding
        training output.

        Args
        ----
        name : string
            Name of the variable output.

        val : float or ndarray
            Initial value for the output. While the value is overwritten during
            execution, it is useful for infering size.

        training_data : float or ndarray
            training data for this variable. Optional, can be set
            by the problem later.
        """
        if training_data is None:
            training_data = []

        super(MetaModel, self).add_output(name, val, **kwargs)
        super(MetaModel, self).add_param('train:'+name, val=training_data, pass_by_obj=True)

        try:
            output_shape = self._init_unknowns_dict[name]['shape']
        except KeyError: #then its some kind of object, and just assume scalar training data
            output_shape = 1

        self._surrogate_output_names.append((name, output_shape))
        self._training_output[name] = np.zeros(0)

        if self._init_unknowns_dict[name].get('surrogate'):
            self._init_unknowns_dict[name]['default_surrogate'] = False
        else:
            self._init_unknowns_dict[name]['default_surrogate'] = True

    def _setup_variables(self, compute_indices=False):
        """Returns our params and unknowns dictionaries,
        re-keyed to use absolute variable names.

        Also instantiates surrogates for the output variables
        that use the default surrogate.

        Args
        ----

        compute_indices : bool, optional
            If True, call setup_distrib() to set values of
            'src_indices' metadata.
        """
        # create an instance of the default surrogate for outputs that
        # did not have a surrogate specified
        if self.default_surrogate is not None:
            for name, shape in self._surrogate_output_names:
                if self._init_unknowns_dict[name].get('default_surrogate'):
                    surrogate = deepcopy(self.default_surrogate)
                    self._init_unknowns_dict[name]['surrogate'] = surrogate

        # training will occur on first execution after setup
        self.train = True

        return super(MetaModel, self)._setup_variables(compute_indices)

    def check_setup(self, out_stream=sys.stdout):
        """Write a report to the given stream indicating any potential problems found
        with the current configuration of this ``MetaModel``.

        Args
        ----
        out_stream : a file-like object, optional
        """

        # All outputs must have surrogates assigned
        # either explicitly or through the default surrogate
        if self.default_surrogate is None:
            no_sur = []
            for name, shape in self._surrogate_output_names:
                surrogate = self._init_unknowns_dict[name].get('surrogate')
                if surrogate is None:
                    no_sur.append(name)
            if len(no_sur) > 0:
                msg = ("No default surrogate model is defined and the following"
                       " outputs do not have a surrogate model:\n%s\n"
                       "Either specify a default_surrogate, or specify a "
                       "surrogate model for all outputs."
                       % no_sur)
                out_stream.write(msg)

    def solve_nonlinear(self, params, unknowns, resids):
        """Predict outputs.
        If the training flag is set, train the metamodel first.

        Args
        ----
        params : `VecWrapper`, optional
            `VecWrapper` containing parameters. (p)

        unknowns : `VecWrapper`, optional
            `VecWrapper` containing outputs and states. (u)

        resids : `VecWrapper`, optional
            `VecWrapper` containing residuals. (r)
        """
        # Train first
        if self.train:
            self._train()

        # Now Predict for current inputs
        inputs = self._params_to_inputs(params)

        for name, shape in self._surrogate_output_names:
            surrogate = self._init_unknowns_dict[name].get('surrogate')
            if surrogate:
                unknowns[name] = surrogate.predict(inputs)
            else:
                raise RuntimeError("Metamodel '%s': No surrogate specified for output '%s'"
                                   % (self.pathname, name))

    def _params_to_inputs(self, params, out=None):
        """
        Converts from a dictionary of parameters to the ndarray input.
        """
        if out is None:
            inputs = np.zeros(self._input_size)
        else:
            inputs = out

        idx = 0
        for name, sz in self._surrogate_param_names:
            val = params[name]
            if isinstance(val, list):
                val = np.array(val)
            if isinstance(val, np.ndarray):
                inputs[idx:idx + sz] = val.flat
                idx += sz
            else:
                inputs[idx] = val
                idx += 1
        return inputs

    def linearize(self, params, unknowns, resids):
        """
        Returns the Jacobian as a dictionary whose keys are tuples of the form
         ('unknown', 'param') and whose values are ndarrays.

        Args
        ----
        params : `VecWrapper`
            `VecWrapper` containing parameters. (p)

        unknowns : `VecWrapper`
            `VecWrapper` containing outputs and states. (u)

        resids : `VecWrapper`
            `VecWrapper` containing residuals. (r)

        Returns
        -------
        dict
            Dictionary whose keys are tuples of the form ('unknown', 'param')
            and whose values are ndarrays.
        """

        jac = {}
        inputs = self._params_to_inputs(params)

        for uname, _ in self._surrogate_output_names:
            surrogate = self._init_unknowns_dict[uname].get('surrogate')
            sjac = surrogate.linearize(inputs)

            idx = 0
            for pname, sz in self._surrogate_param_names:
                jac[(uname, pname)] = sjac[:, idx:idx+sz]
                idx += sz

        return jac

    def _train(self):
        """
        Train the metamodel, if necessary, using the provided training data.
        """

        num_sample = None
        for name, sz in self._surrogate_param_names:
            val = self.params['train:' + name]
            if num_sample is None:
                num_sample = len(val)
            elif len(val) != num_sample:
                msg = "MetaModel: Each variable must have the same number"\
                      " of training points. Expected {0} but found {1} "\
                      "points for '{2}'."\
                      .format(num_sample, len(val), name)
                raise RuntimeError(msg)

        for name, shape in self._surrogate_output_names:
            val = self.params['train:' + name]
            if len(val) != num_sample:
                msg = "MetaModel: Each variable must have the same number" \
                      " of training points. Expected {0} but found {1} " \
                      "points for '{2}'." \
                    .format(num_sample, len(val), name)
                raise RuntimeError(msg)

        if self.warm_restart:
            num_old_pts = self._training_input.shape[0]
            inputs = np.zeros((num_sample + num_old_pts, self._input_size))
            if num_old_pts > 0:
                inputs[:num_old_pts, :] = self._training_input
            new_input = inputs[num_old_pts:, :]

        else:
            inputs = np.zeros((num_sample, self._input_size))
            new_input = inputs

        self._training_input = inputs

        # add training data for each input
        if num_sample > 0:
            idx = 0
            for name, sz in self._surrogate_param_names:
                val = self.params['train:' + name]
                if isinstance(val[0], float):
                    new_input[:, idx] = val
                    idx += 1
                else:
                    for row_idx, v in enumerate(val):
                        if not isinstance(v, np.ndarray):
                            v = np.array(v)
                        new_input[row_idx, idx:idx+sz] = v.flat

        # add training data for each output
        for name, shape in self._surrogate_output_names:
            if num_sample > 0:
                output_size = np.prod(shape)

                if self.warm_restart:
                    outputs = np.zeros((num_sample + num_old_pts,
                                        output_size))
                    if num_old_pts > 0:
                        outputs[:num_old_pts, :] = self._training_output[name]
                    self._training_output[name] = outputs
                    new_output = outputs[num_old_pts:, :]
                else:
                    outputs = np.zeros((num_sample, output_size))
                    self._training_output[name] = outputs
                    new_output = outputs

                val = self.params['train:' + name]

                if isinstance(val[0], float):
                    new_output[:, 0] = val
                else:
                    for row_idx, v in enumerate(val):
                        if not isinstance(v, np.ndarray):
                            v = np.array(v)
                        new_output[row_idx, :] = v.flat

            surrogate = self._init_unknowns_dict[name].get('surrogate')
            if surrogate is not None:
                surrogate.train(self._training_input, self._training_output[name])

        self.train = False

    def _get_fd_params(self):
        """
        Get the list of parameters that are needed to perform a
        finite difference on this `Component`.

        Returns
        -------
        list of str
            List of names of params for this `Component` .
        """
        return [k for k, acc in iteritems(self.params._dat)
                   if not (acc.pbo or k.startswith('train'))]

    def _get_fd_unknowns(self):
        """
        Get the list of unknowns that are needed to perform a
        finite difference on this `Component`.

        Returns
        -------
        list of str
            List of names of unknowns for this `Component`.
        """
        return [k for k, acc in iteritems(self.unknowns._dat)
                   if not (acc.pbo or k.startswith('train'))]
