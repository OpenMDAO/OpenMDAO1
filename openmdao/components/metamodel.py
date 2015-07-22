""" Metamodel provides basic Meta Modeling capability."""

import sys
from copy import deepcopy

from openmdao.core.component import Component, _NotSet


class MetaModel(Component):
    """ Class that creates a reduced order model for a tuple of outputs from
    a tuple of inputs. Accepts surrogate models that adhere to ISurrogate.
    Multiple surrogate models can be used. Training inputs and outputs should
    be provided in the params and responses variable trees.

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
        self._train = True

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
        super(MetaModel, self).add_output('train:'+name,  val=list(), pass_by_obj=True)
        self._surrogate_output_names.append(name)

    def _setup_variables(self):
        """ Returns our params and unknowns dictionaries,
        re-keyed to use absolute variable names, and stores
        them as attributes of the component.

        Also instantiates surrogates for the output variables
        that use the default surrogate.
        """
        _new_params, _new_unknowns = super(MetaModel, self)._setup_variables()

        # create an instance of the default surrogate for outputs that do not
        # already have a surrogate in their metadata
        if self.default_surrogate is not None:
            for name in self._surrogate_output_names:
                surrogate = self._unknowns_dict[name].get('surrogate')
                if surrogate is None:
                    surrogate = deepcopy(self.default_surrogate)
                    self._unknowns_dict[name]['surrogate'] = surrogate

        return _new_params, _new_unknowns

    def _input_updated(self, name, fullpath=None):
        """ Set _train if anything changes in our inputs so that training
        occurs on the next execution."""

        if fullpath is not None:
            if fullpath.startswith('params.') or \
               fullpath.startswith('responses.'):
                self._train = True

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
                print('check_setup: surrogate for', name, 'is', surrogate)
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
        """
        If the training flag is set, train the metamodel. Otherwise,
        predict outputs.

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
        if self._train:
            print('\n%s Training...' % self.name)
            if self.warm_restart:
                base = len(input_data)
            else:
                input_data = []
                base = 0

            # add training data for each input
            for name in self._surrogate_param_names:
                val = self.params['train:'+name]
                num_sample = len(val)

                for j in range(base, base + num_sample):
                    if j > len(input_data) - 1:
                        input_data.append([])
                    input_data[j].append(val[j-base])

            # Surrogate models take an (m, n) list of lists
            # m = number of training samples
            # n = number of inputs
            #
            # TODO - Why not numpy array instead?

            # add training data for each output
            for name in self._surrogate_output_names:
                if not self.warm_restart:
                    output_data = []

                output_data.extend(self.unknowns['train:'+name])
                surrogate = self._unknowns_dict[name].get('surrogate')

                print('\ninput_data:\n', input_data)
                print('\noutput_data:\n', output_data)
                if surrogate is not None:
                    surrogate.train(input_data, output_data)

            self._train = False

        # Now Predict for current inputs
        print('\n%s Predicting...' % self.name)
        inputs = []
        for name in self._surrogate_param_names:
            val = params[name]
            inputs.append(val)

        for name in self._surrogate_output_names:
            surrogate = self._unknowns_dict[name].get('surrogate')
            if surrogate is not None:
                unknowns[name] = surrogate.predict(inputs)
            else:
                raise RuntimeError("Metamodel '%s': No surrogate specified for output '%s'"
                                   % (self.pathname, name))
