class OptionsDictionary(object ):
    def __init__(self):
        self._options = {}
        
    def add_option(self, name, value, **kargs):
        self._options[name] = value
        
    def __getitem__(self, name):
        try:
            return self._options[name]
        except KeyError as error:
            raise ValueError("'{}' is not a valid option".format(name))
            
    def __setitem__(self, name, value):
        self._options[name] = value
        
        
        
