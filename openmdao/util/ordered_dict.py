
from six.moves import range, zip, _dummy_thread
from six import iteritems

try:
    from thread import get_ident as _get_ident
except ImportError:
    from _dummy_thread import get_ident as _get_ident


class OrderedDict(dict):
    """
    An alternative to built-in OrderedDict.  This one uses a list to keep
    the value order and a map of keys to index into the list.  It will be much
    slower to delete entries, but should be faster for iteration.
    """
    def __init__(*args, **kwargs):
        if not args:
            raise TypeError("descriptor '__init__' of 'OrderedDict' object "
                            "needs an argument")
        self = args[0]
        args = args[1:]
        if len(args) > 1:
            raise TypeError('expected at most 1 arguments, got %d' % len(args))

        self._map = {}  # keys mapped to index into lists
        self._vallist = []    # list of values
        self._keylist = []    # list of keys

        self.update(*args, **kwargs)

    def update(self, *args, **kwargs):
        if len(args) == 1:
            if hasattr(args[0], 'keys'):
                for k in args[0]:
                    self[k] = args[0][k]
            else:
                for k, v in args[0]:
                    self[k] = v
            for k,v in kwargs.items():
                self[k] = v
        elif len(args) > 1:
            raise TypeError('expected at most 1 arguments, got %d' % len(args))

    def __len__(self):
        return len(self._keylist)

    def get(self, key, default=None):
        if key in self._map:
            return self._vallist[self._map[key]]
        return default

    def __getitem__(self, key):
        return self._vallist[self._map[key]]

    def __setitem__(self, key, value):
        if key not in self._map:
            self._map[key] = len(self._vallist)
            self._vallist.append(value)
            self._keylist.append(key)
        else:
            self._vallist[self._map[key]] = value

    def __delitem__(self, key):
        idx = self._map[key]
        self._vallist.pop(idx)
        self._keylist.pop(idx)
        del self._map[key]
        for k, i in iteritems(self._map):
            if i > idx:
                self._map[k] -= 1

    def __contains__(self, key):
        return key in self._map

    def __iter__(self):
        return iter(self._keylist)

    def __reversed__(self):
        return reversed(self._keylist)

    def clear(self):
        self._map = {}
        self._vallist = []
        self._keylist = []

    def keys(self):
        return self._keylist

    def values(self):
        return self._vallist

    def items(self):
        return zip(self._keylist, self._vallist)

    def iterkeys(self):
        return iter(self._keylist)

    def itervalues(self):
        return iter(self._vallist)

    def iteritems(self):
        return zip(self._keylist, self._vallist)

    __marker = object()

    def pop(self, key, default=__marker):
        '''od.pop(k[,d]) -> v, remove specified key and return the corresponding
        value.  If key is not found, d is returned if given, otherwise KeyError
        is raised.

        '''
        if key in self:
            result = self[key]
            del self[key]
            return result
        if default is self._marker:
            raise KeyError(key)
        return default

    def setdefault(self, key, default=None):
        'od.setdefault(k[,d]) -> od.get(k,d), also set od[k]=d if k not in od'
        if key in self:
            return self[key]
        self[key] = default
        return default

    def popitem(self, last=True):
        '''od.popitem() -> (k, v), return and remove a (key, value) pair.
        Pairs are returned in LIFO order if last is true or FIFO order if false.

        '''
        if not self:
            raise KeyError('dictionary is empty')
        key = next(reversed(self) if last else iter(self))
        value = self.pop(key)
        return key, value

    def __repr__(self, _repr_running={}):
        'od.__repr__() <==> repr(od)'
        call_key = id(self), _get_ident()
        if call_key in _repr_running:
            return '...'
        _repr_running[call_key] = 1
        try:
            if not self:
                return '%s()' % (self.__class__.__name__,)
            return '%s(%r)' % (self.__class__.__name__, self.items())
        finally:
            del _repr_running[call_key]

    def __reduce__(self):
        'Return state information for pickling'
        items = [[k, self[k]] for k in self]
        inst_dict = vars(self).copy()
        for k in vars(OrderedDict()):
            inst_dict.pop(k, None)
        if inst_dict:
            return (self.__class__, (items,), inst_dict)
        return self.__class__, (items,)

    def copy(self):
        'od.copy() -> a shallow copy of od'
        return self.__class__(self)

    @classmethod
    def fromkeys(cls, iterable, value=None):
        '''OD.fromkeys(S[, v]) -> New ordered dictionary with keys from S.
        If not specified, the value defaults to None.

        '''
        self = cls()
        for key in iterable:
            self[key] = value
        return self

    def __eq__(self, other):
        '''od.__eq__(y) <==> od==y.  Comparison to another OD is order-sensitive
        while comparison to a regular mapping is order-insensitive.

        '''
        if isinstance(other, OrderedDict):
            return dict._eq__(self, other) and all(_imap(_eq, self, other))
        return dict._eq__(self, other)

    def __ne__(self, other):
        'od.__ne__(y) <==> od!=y'
        return not self == other

    # -- the following methods support python 3.x style dictionary views --

    def viewkeys(self):
        "od.viewkeys() -> a set-like object providing a view on od's keys"
        return KeysView(self)

    def viewvalues(self):
        "od.viewvalues() -> an object providing a view on od's values"
        return ValuesView(self)

    def viewitems(self):
        "od.viewitems() -> a set-like object providing a view on od's items"
        return ItemsView(self)

from collections import OrderedDict
