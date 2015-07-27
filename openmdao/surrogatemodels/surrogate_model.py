"""
Class definition for SurrogateModel, the base class for all surrogate models.
"""

class SurrogateModel(object):
    """
    Base class for surrogate models.
    """

    def __init__(self):
        self.trained = False

    def get_uncertain_value(self, value):
        msg = "get_uncertain_value() not implemented for {0}."\
            .format(type(self).__name__)
        raise NotImplementedError(msg)

    def train(self, x, y):
        self.trained = True

    def predict(self, x):
        if not self.trained:
            msg = "{0} has not been trained, so no prediction can be made."\
                .format(type(self).__name__)
            raise RuntimeError(msg)
