
from collections import OrderedDict
from openmdao.core.system import System


class Component(System):
    """A System that is responsible for creating variables"""

    def __init__(self):
        super(Component, self).__init__()
        self._post_setup = False

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

    def _check_name(self, name):
        if self._post_setup:
            raise RuntimeError("%s: can't add variable %s because setup has already been called",
                               (self.pathname, name))
        if name in self._params or name in self._unknowns:
            raise RuntimeError("%s: variable %s already exists" %
                               (self.pathname, name))

    def setup_variables(self):
        """Returns our params and unknowns, and stores them
        as attributes of the component"""

        # rekey with absolute path names and add relative names

        _new_params = OrderedDict()
        for name, meta in self._params.items():
            if not self.pathname:
                var_pathname = name
            else:
                var_pathname = ':'.join([self.pathname, name])
            _new_params[var_pathname] = meta
            meta['relative_name'] = name
        self._params = _new_params

        _new_unknowns = OrderedDict()
        for name, meta in self._unknowns.items():
            if not self.pathname:
                var_pathname = name
            else:
                var_pathname = ':'.join([self.pathname, name])
            _new_unknowns[var_pathname] = meta
            meta['relative_name'] = name
        self._unknowns = _new_unknowns

        self._post_setup = True

        return self._params, self._unknowns
