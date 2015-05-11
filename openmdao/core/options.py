
class OptionsDictionary(object ):
    def __init__(self):
        self._options = {}

    def add_option(self, name, value, low=None, high=None, values=None):

        if name in self._options:
            raise ValueError("Option '{}' already exists".format(name))

        self._options[name] = {
            'val':    value,
            'low':    low,
            'high':   high,
            'values': values,
        }

        self.check(name, value)

    def __getitem__(self, name):
        try:
            return self._options[name]['val']
        except KeyError:
            raise KeyError("Option '{}' has not been added".format(name))

    def __setitem__(self, name, value):
        if name not in self._options:
            raise KeyError("Option '{}' has not been added".format(name))

        self.check(name, value)
        self._options[name]['val'] = value

    def check(self, name, value):
        low    = self._options[name]['low']
        high   = self._options[name]['high']
        values = self._options[name]['values']
        _type  = type(self._options[name]['val'])

        self._check_type(name, value, _type)

        if low is not None:
            self._check_low(name, value, low)

        if high is not None:
            self._check_high(name, value, high)

        if values is not None:
            self._check_values(name, value, values)

    def _check_type(self, name, value, _type):
        if type(value) != _type:
            msg = "'{}' should be a '{}'"
            raise ValueError(msg.format(name, _type))

    def _check_low(self, name, value, low):
        if value < low:
            msg = "minimum allowed value for '{}' is '{}'"
            raise ValueError(msg.format(name, low))

    def _check_high(self, name, value, high):
        if value > high:
            msg = "maximum allowed value for '{}' is '{}'"
            raise ValueError(msg.format(name, high))

    def _check_values(self, name, value, values):
        if value not in values:
            msg = "'{}' must be one of the following values: '{}'"
            raise ValueError(msg.format(name, values))
