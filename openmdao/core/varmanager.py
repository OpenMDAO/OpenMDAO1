
class VarManager(object):
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
