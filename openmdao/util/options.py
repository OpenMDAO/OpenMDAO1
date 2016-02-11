""" OptionsDictionary class definition. """
from copy import deepcopy
from six import iteritems

class OptionsDictionary(object):
    """ A dictionary for storing options for components/drivers/solvers. It
    is generally used like a standard Python dictionary, except that 1) you
    can only set or get keys that have been registered with add_option, and
    2) type is enforced

    Args
    ----
    read_only : bool
        If this is True, these options should not be modified at run time,
        and should not be printed in the docs.
    ."""

    def __init__(self, read_only=True):
        self._options = {}
        self.read_only = read_only

    def add_option(self, name, value, lower=None, upper=None, values=None,
                   desc=''):
        """ Adds an option to this options dictionary.

        Args
        ----
        name : str
            Name of the option.

        value : object
            Default value for this option. The type of this value will be enforced.

        lower : float, optional
            Lower bounds for a float value.

        upper : float, optional
            Upper bounds for a float value.

        values : list, optional
            List of all possible values for an enumeration option.

        desc : str, optional
            String containing documentation of this option.
        """
        if name in self._options:
            print("raising an error")
            raise ValueError("Option '{}' already exists".format(name))

        self._options[name] = {
            'val':    value,
            'lower':    lower,
            'upper':   upper,
            'values': values,
            'desc' : desc,
        }

        self._check(name, value)

    def __getitem__(self, name):
        try:
            return self._options[name]['val']
        except KeyError:
            raise KeyError("Option '{}' has not been added".format(name))

    def __contains__(self, name):
        return name in self._options

    def __setitem__(self, name, value):
        if name not in self._options:
            raise KeyError("Option '{}' has not been added".format(name))

        self._check(name, value)
        self._options[name]['val'] = value

    def __setattr__(self, name, value):
        """ To prevent user error, disallow direct setting."""
        if name in ['_options', 'read_only']:
            super(OptionsDictionary, self).__setattr__(name, value)
        else:
            raise ValueError("Use dict-like access for option '{}'".format(name))

    def get(self, name, default=None):
        """
        Returns
        -------
        object
            The value of the named option.  If not found, returns the
            default value that was passed in.
        """
        if name in self._options:
            return self._options[name]['val']
        return default

    def iteritems(self):
        return self.items()

    def items(self):
        """
        Returns
        -------
        iterator
            Iterator returning the name and option for each option.
        """
        return ((name, opt['val']) for name, opt in iteritems(self._options))

    def get_desc(self, name):
        return self._options[name]['desc']

    def _check(self, name, value):
        """ Type checking happens here. """
        lower = self._options[name]['lower']
        upper = self._options[name]['upper']
        values = self._options[name]['values']
        _type = type(self._options[name]['val'])

        self._check_type(name, value, _type)

        if lower is not None:
            self._check_low(name, value, lower)

        if upper is not None:
            self._check_high(name, value, upper)

        if values is not None:
            self._check_values(name, value, values)

    def _check_type(self, name, value, _type):
        """ Check for type. """
        if type(value) != _type:
            msg = "'{}' should be a '{}'"
            raise ValueError(msg.format(name, _type))

    def _check_low(self, name, value, lower):
        """ Check for violation of lower bounds. """
        if value < lower:
            msg = "minimum allowed value for '{}' is '{}'"
            raise ValueError(msg.format(name, lower))

    def _check_high(self, name, value, upper):
        """ Check for violation of upper bounds. """
        if value > upper:
            msg = "maximum allowed value for '{}' is '{}'"
            raise ValueError(msg.format(name, upper))

    def _check_values(self, name, value, values):
        """ Check for value not in enumeration. """
        if value not in values:
            msg = "'{}' must be one of the following values: '{}'"
            raise ValueError(msg.format(name, values))

    def _generate_docstring(self, dictname):
        """
        Generates a numpy-style docstring for an OptionsDictionary.

        Returns
        -------
        docstring : str
            string that contains part of a basic numpy docstring.

        """
        docstring = []
        for (name, val) in sorted(self.items()):
            docstring.extend(["    ", dictname, "['", name, "']",
                                " :  ", type(val).__name__, "("])
            if isinstance(val, str):
                docstring.append("'%s'"%val)
            else:
                docstring.append(str(val))
            docstring.append(")\n")

            desc = self._options[name]['desc']
            if(desc):
                docstring.extend(["        ", desc, "\n"])

        return ''.join(docstring)
