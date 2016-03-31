""" OptionsDictionary class definition. """

import warnings
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
        and should not be printed in the docs."""

    # When this is True, variables marked as 'lock_on_setup' cannot be
    # changed. In all of OpenMDAO's default OptDicts, this will be set to
    # True on setup.
    locked = False

    def __init__(self, read_only=True):
        self._options = {}
        self._deprecations = {}

        # When this is True, no variables in the dictionary can be modified.
        self.read_only = read_only

        # Start out with all dictionaries unlocked. Nobody should be creating
        # them after setup.
        # TODO: This really is a hack, but we couldn't figure out a way
        # around it.
        OptionsDictionary.locked = False

    def add_option(self, name, value, lower=None, upper=None, values=None,
                   desc='', lock_on_setup=False):
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

        opt = {
            'val': value,
            'lower' : lower,
            'upper' : upper,
            'values' : values,
            'desc' : desc,
            'lock_on_setup' : lock_on_setup
        }

        self._check(name, value, opt)

        self._options[name] = opt

    def remove_option(self, name):
        """ Removes the named option.  Does nothing if the option is not found.

        Args
        ----
        name : str
            Name of the option to remove.

        """
        try:
            del self._options[name]
        except KeyError:
            pass

    def _add_deprecation(self, oldname, newname):
        """ For renamed options, maps the old name to the new name and prints
        a DeprecationWarning on each get/set that uses the old name.

        Args
        ----
        oldname : str
            The deprecated name.

        newname : str
            The correct name.
        """
        if newname not in self._options:
            raise NameError("The '%s' option was not found." % newname)
        self._deprecations[oldname] = newname

    def __getitem__(self, name):
        try:
            return self._options[name]['val']
        except KeyError:
            try:
                newname = self._deprecations[name]
                _print_deprecation(name, newname)
                return self._options[newname]['val']
            except KeyError:
                raise KeyError("Option '{}' has not been added".format(name))

    def __contains__(self, name):
        return name in self._options or name in self._deprecations

    def __setitem__(self, name, value):
        """ Set an option using dictionary-like access."""

        if name not in self._options:
            if name in self._deprecations:
                newname = self._deprecations[name]
                _print_deprecation(name, newname)
                name = newname
            else:
                raise KeyError("Option '{}' has not been added".format(name))

        opt = self._options[name]
        self._check(name, value, opt)
        opt['val'] = value

    def __setattr__(self, name, value):
        """ To prevent user error, disallow direct setting."""
        if name in ['_options', 'read_only', '_deprecations', 'locked']:
            super(OptionsDictionary, self).__setattr__(name, value)
        else:
            raise ValueError("Use dict-like access for option '{}'".format(name))

    def get(self, name, default=None):
        """ Gets a value from this OptionsDictionary.

        Returns
        -------
        object
            The value of the named option.  If not found, returns the
            default value that was passed in.
        """
        if name in self._options:
            return self._options[name]['val']
        elif name in self._deprecations:
            newname = self._deprecations[name]
            _print_deprecation(name, newname)
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

    def _check(self, name, value, opt):
        """ Type checking happens here. """

        # Raise an error when trying to set a restricted variable after
        # setup.
        if self.locked and opt['lock_on_setup']:
            msg = "The '%s' option cannot be changed after setup." % name
            raise RuntimeError(msg)

        values = opt['values']

        if values is not None:
            # Only need to check if we are in the list if we are an enum
            self._check_values(name, value, values)

        else:
            lower = opt['lower']
            upper = opt['upper']
            _type = type(opt['val'])

            self._check_type(name, value, _type)

            if lower is not None:
                self._check_low(name, value, lower)

            if upper is not None:
                self._check_high(name, value, upper)

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
        """ Generates a numpy-style docstring for an OptionsDictionary.

        Returns
        -------
        docstring : str
            string that contains part of a basic numpy docstring.

        """
        docstring = []
        for (name, val) in sorted(self.items()):
            docstring.extend(["    ", dictname, "['", name, "']",
                                " : ", type(val).__name__, "("])
            if isinstance(val, str):
                docstring.append("'%s'"%val)
            else:
                docstring.append(str(val))
            docstring.append(")\n")

            desc = self._options[name]['desc']
            if(desc):
                docstring.extend(["        ", desc, "\n"])

        return ''.join(docstring)

def _print_deprecation(name, newname):
    warnings.simplefilter('always', DeprecationWarning)
    warnings.warn("Option '%s' is deprecated. Use '%s' instead." %
                  (name, newname),
                  DeprecationWarning,stacklevel=2)
    warnings.simplefilter('ignore', DeprecationWarning)
