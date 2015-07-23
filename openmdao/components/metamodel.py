""" Metamodel provides basic Meta Modeling capability."""

import sys
from copy import deepcopy

from openmdao.core.component import Component, _NotSet


class MetaModel(Component):
    """Class that creates a reduced order model for outputs from
    parameters. Each output may have it's own surrogate model.
    Training inputs and outputs are automatically created with
    'train:' prepended to the corresponding parameter/output name.

    For a Float variable, the training data is an array of length m.
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
        self._training_input = []
        self._training_output = {}

        # When set to False (default), the metamodel retrains with the new
        # dataset whenever the training data values are changed. When set to
        # True, the new data is appended to the old data and all of the data
        # is used to train.
        self.warm_restart = False

        # keeps track of which sur_<name> slots are full
        self._surrogate_overrides = set()

    def add_param(self, name, val=_NotSet, **kwargs):
        """ Add a `param` input to this component and a corresponding
        training parameter.

        Args
        ----
        name : string
            Name of the input.

        val : float or ndarray or object
            Initial value for the input.
        """
        super(MetaModel, self).add_param(name, val, **kwargs)
        super(MetaModel, self).add_param('train:'+name, val=list(), pass_by_obj=True)
        self._surrogate_param_names.append(name)

    def add_output(self, name, val=_NotSet, **kwargs):
        """ Add an output to this component and a corresponding
        training output.

        Args
        ----
        name : string
            Name of the variable output.

        val : float or ndarray
            Initial value for the output. While the value is overwritten during
            execution, it is useful for infering size.
        """
        super(MetaModel, self).add_output(name, val, **kwargs)
        super(MetaModel, self).add_output('train:'+name, val=list(), pass_by_obj=True)
        self._surrogate_output_names.append(name)
        self._training_output[name] = []

        if self._unknowns_dict[name].get('surrogate'):
            self._unknowns_dict[name]['default_surrogate'] = False
        else:
            self._unknowns_dict[name]['default_surrogate'] = True

    def _setup_variables(self):
        """Returns our params and unknowns dictionaries,
        re-keyed to use absolute variable names.

        Also instantiates surrogates for the output variables
        that use the default surrogate.
        """
        # create an instance of the default surrogate for outputs that do not
        # already have a surrogate in their metadata
        if self.default_surrogate is not None:
            for name in self._surrogate_output_names:
                if self._unknowns_dict[name].get('default_surrogate'):
                    surrogate = deepcopy(self.default_surrogate)
                    self._unknowns_dict[name]['surrogate'] = surrogate

        # training will occur on first execution after setup
        self.train = True

        return super(MetaModel, self)._setup_variables()

    def check_setup(self, out_stream=sys.stdout):
        """Write a report to the given stream indicating any potential problems found
        with the current configuration of this ``Problem``.

        Args
        ----
        out_stream : a file-like object, optional
        """

        # Either there are no surrogates set and no default surrogate (just
        # do passthrough ) or all outputs must have surrogates assigned
        # either explicitly or through the default surrogate
        if self.default_surrogate is None:
            no_sur = []
            for name in self._surrogate_output_names:
                surrogate = self._unknowns_dict[name].get('surrogate')
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
            if self.warm_restart:
                base = len(self._training_input)
            else:
                self._training_input = []
                base = 0

            # add training data for each input
            for name in self._surrogate_param_names:
                val = self.params['train:'+name]
                num_sample = len(val)

                for j in range(base, base + num_sample):
                    if j > len(self._training_input) - 1:
                        self._training_input.append([])
                    self._training_input[j].append(val[j-base])

            # add training data for each output
            for name in self._surrogate_output_names:
                if not self.warm_restart:
                    self._training_output[name] = []

                self._training_output[name].extend(self.unknowns['train:'+name])
                surrogate = self._unknowns_dict[name].get('surrogate')
                if surrogate is not None:
                    surrogate.train(self._training_input, self._training_output[name])

            self.train = False

        # Now Predict for current inputs
        inputs = []
        for name in self._surrogate_param_names:
            val = params[name]
            inputs.append(val)

        for name in self._surrogate_output_names:
            surrogate = self._unknowns_dict[name].get('surrogate')
            if surrogate:
                unknowns[name] = surrogate.predict(inputs)
            else:
                raise RuntimeError("Metamodel '%s': No surrogate specified for output '%s'"
                                   % (self.pathname, name))
