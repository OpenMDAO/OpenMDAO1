"""Surrogate Model based on second order response surface equations."""

from numpy import linalg, zeros, array

from openmdao.surrogate_models.surrogate_model import SurrogateModel
from six.moves import range


class ResponseSurface(SurrogateModel):
    def __init__(self, X=None, Y=None):
        # must call HasTraits init to set up Traits stuff
        super(ResponseSurface, self).__init__()

        self.m = 0  # number of training points
        self.n = 0  # number of independents
        self.betas = None  # vector of response surface equation coefficients

        if X is not None and Y is not None:
            self.train(X, Y)

    def train(self, x, y):
        """ Calculate response surface equation coefficients using least squares regression. """
        
        super(ResponseSurface, self).train(x, y)

        x = array(x)
        y = array(y).T

        m = self.m = x.shape[0]
        n = self.n = x.shape[1]

        X = zeros((m, ((n + 1) * (n + 2)) // 2))

        # Modify X to include constant, squared terms and cross terms

        # Constant Terms
        X[:, 0] = 1.0

        # Linear Terms
        X[:, 1:n+1] = x

        # Quadratic Terms
        X_offset = X[:, n + 1:]
        idx = 0
        for i in range(n):
            for j in range(i, n):
                X_offset[:, idx] = x[:, i] * x[:, j]
                idx += 1

        # Determine response surface equation coefficients (betas) using least squares
        self.betas, rs, r, s = linalg.lstsq(X, y)

    def predict(self, x):
        """Calculates a predicted value of the response based on the current response surface model for the supplied list of inputs. """

        super(ResponseSurface, self).predict(x)

        x = array(x)
        if len(x.shape) == 1:
            x.shape = (1, len(x))
        m = x.shape[0]
        n = x.shape[1]

        X = zeros((m, ((self.n + 1) * (self.n + 2)) // 2))

        # Modify X to include constant, squared terms and cross terms

        # Constant Terms
        X[:, 0] = 1.0

        # Linear Terms
        X[:, 1:n + 1] = x

        # Quadratic Terms
        X_offset = X[:, self.n + 1:]
        idx = 0
        for i in range(self.n):
            for j in range(i, self.n):
                X_offset[:, idx] = x[:, i] * x[:, j]
                idx += 1

        # Predict new_y using new_x and betas
        y = X.dot(self.betas)
        return y[0]
