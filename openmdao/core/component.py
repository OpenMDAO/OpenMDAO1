
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

    def setup_variables(self):
        """Returns our params and unknowns, and stores them
        as attributes of the component"""
        
        # rekey with absolute path names and add relative names
        
        _new_params = OrderedDict()
        for name, meta in self._params.items():
            var_pathname = ':'.join([self.pathname, name])
            _new_params[var_pathname] = meta
            meta['relative_name'] = name
        self._params = _new_params

        _new_unknowns = OrderedDict()
        for name, meta in self._unknowns.items():
            var_pathname = ':'.join([self.pathname, name])
            _new_unknowns[var_pathname] = meta
            meta['relative_name'] = name
        self._unknowns = _new_unknowns
        
        return self._params, self._unknowns
