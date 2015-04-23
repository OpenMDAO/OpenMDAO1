
from collections import OrderedDict
from openmdao.core.system import System


class Component(System):
    def __init__(self):
        super(Component, self).__init__()
        self._params = OrderedDict()
        self._unknowns = OrderedDict()
        self._states = OrderedDict()

        # by default, don't promote any vars up to our parent
        self.promotes = ()

    def add_param(self, name, val, **kwargs):
        args = kwargs.copy()
        args['val'] = val
        self._params[name] = args

    def add_unknown(self, name, val, **kwargs):
        args = kwargs.copy()
        args['val'] = val
        self._unknowns[name] = args

    def add_state(self, name, val, **kwargs):
        args = kwargs.copy()
        args['val'] = val
        self._states[name] = args

    def variables(self):
        return self._params, self._unknowns, self._states

