
class VarManager(object):
    """A manager of the data transfer of a possibly distributed
    collection of variables.
    """
    def __init__(self):
        pass

    def __getitem__(self, name):
        """Retrieve unflattened value of named var."""
        pass

    def __setitem__(self, name, value):
        """Set the value of the named var"""
        pass

    def flat(self, name):
        """Retrieve flattened value of named var.
        Raises exception if value is nonflattenable.
        """
        pass
