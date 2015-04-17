
class VarManager(object):
    def __getitem__(self, name):
        """Retrieve unflattened value of named var."""
        pass

    def flat(self, name):
        """Retrieve flattened value of named var.
        Raises exception if value is nonflattenable.
        """
        pass
