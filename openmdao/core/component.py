
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
        args['state'] = True
        self._states[name] = args

    def setup_paths(self, parent_path):
        """Set the absolute pathname of each Variable in the
        Component.
        """
        super(Component, self).setup_paths(parent_path)

        for name, meta in self._params.items():
            meta['pathname'] = ':'.join((self.pathname, name))
        for name, meta in self._outputs.items():
            meta['pathname'] = ':'.join((self.pathname, name))
        for name, meta in self._states.items():
            meta['pathname'] = ':'.join((self.pathname, name))

    def variables(self):
        unknowns = OrderedDict()
        unknowns.update(self._states)
        unknowns.update(self._outputs)
        return self._params, unknowns
