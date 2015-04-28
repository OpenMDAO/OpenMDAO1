
from collections import OrderedDict
from openmdao.core.system import System


class Component(System):
    def __init__(self):
        super(Component, self).__init__()
        self._params = OrderedDict()
        self._outputs = OrderedDict()
        self._states = OrderedDict()

    def add_param(self, name, val, **kwargs):
        args = kwargs.copy()
        args['val'] = val
        self._params[name] = args

    def add_output(self, name, val, **kwargs):
        args = kwargs.copy()
        args['val'] = val
        self._outputs[name] = args

    def add_state(self, name, val, **kwargs):
        args = kwargs.copy()
        args['val'] = val
        self._states[name] = args

    def variables(self):
        return self._params, self._outputs, self._states
