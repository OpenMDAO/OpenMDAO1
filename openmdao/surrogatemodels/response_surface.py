"""Surrogate Model based on second order response surface equations."""

from numpy import matrix, linalg, power, multiply, concatenate, ones


class ResponseSurface(object):
    def __init__(self, X=None, Y=None):
        # must call HasTraits init to set up Traits stuff
        super(ResponseSurface, self).__init__()

        self.m = None  # number of training points
        self.n = None  # number of independents
        self.betas = None  # vector of response surface equation coefficients

        if X is not None and Y is not None:
            self.train(X, Y)

    def get_uncertain_value(self, value):
        """Returns the value iself. Response surface equations don't have uncertainty."""
        return value

    def train(self, X, Y):
        """ Calculate response surface equation coefficients using least squares regression. """

        X = matrix(X)
        Y = matrix(Y).T

        self.m = X.shape[0]
        self.n = X.shape[1]

        # Modify X to include constant, squared terms and cross terms
        X = concatenate((matrix(ones((self.m, 1))), X), 1)
        for i in range(1, self.n + 1):
            X = concatenate((X, power(X[:, i], 2)), 1)
        for i in range(1, self.n):
            for j in range(i + 1, self.n + 1):
                X = concatenate((X, multiply(X[:, i], X[:, j])), 1)

        # Determine response surface equation coefficients (betas) using least squares
        self.betas, rs, r, s = linalg.lstsq(X, Y)

    def predict(self, new_x):
        """Calculates a predicted value of the response based on the current response surface model for the supplied list of inputs. """

        new_x = matrix(new_x)

        # Modify new_x to include constant, squared terms and cross terms
        new_x = concatenate((matrix(ones((1, 1))), new_x), 1)
        for i in range(1, self.n + 1):
            new_x = concatenate((new_x, power(new_x[:, i], 2)), 1)
        for i in range(1, self.n):
            for j in range(i + 1, self.n + 1):
                new_x = concatenate((new_x, multiply(new_x[:, i], new_x[:, j])), 1)

        # Predict new_y using new_x and betas
        new_y = new_x * self.betas
        return new_y[0, 0]
