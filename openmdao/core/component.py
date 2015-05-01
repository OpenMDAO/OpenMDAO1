
from collections import OrderedDict
from openmdao.core.system import System


class Component(System):
    """A System that is responsible for creating variables"""
    
    def add_param(self, name, val, **kwargs):
        args = kwargs.copy()
        args['val'] = val
        self._params[name] = args

    def add_output(self, name, val, **kwargs):
        args = kwargs.copy()
        args['val'] = val
        self._unknowns[name] = args

    def add_state(self, name, val, **kwargs):
        args = kwargs.copy()
        args['val'] = val
        args['state'] = True
        self._unknowns[name] = args

    def setup_paths(self, parent_path):
        """Set the absolute pathname of each Variable in the
        Component.
        """
        super(Component, self).setup_paths(parent_path)

        for name, meta in self._params.items():
            meta['pathname'] = ':'.join((self.pathname, name))
        for name, meta in self._unknowns.items():
            meta['pathname'] = ':'.join((self.pathname, name))

    def setup_variables():
        """Returns our params and unknowns"""
        return self._params, self._unknowns