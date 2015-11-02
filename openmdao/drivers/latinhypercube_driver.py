"""
OpenMDAO design-of-experiments driver implementing the Latin Hypercube method.
"""

from openmdao.drivers.predeterminedruns_driver import PredeterminedRunsDriver
from six import moves, iteritems
from random import shuffle
import numpy as np

def rand_latin_hypercube(n, k):
    """
    Calculates a random Latin hypercube set of n points in k
    dimensions within [0,n-1]^k hypercube.
    n: int
       Desired number of points.
    k: int
       Number of design variables (dimensions).
    """
    X = np.zeros((n, k))
    row = range(0, n)
    for i in range(k):
        shuffle(row)
        X[:,i] = row
    return X

class LatinHypercubeDriver(PredeterminedRunsDriver):
    def __init__(self, num_samples=1):
        super(LatinHypercubeDriver, self).__init__()
        self.num_samples = num_samples

    def _build_runlist(self):
        design_vars = self.get_desvar_metadata()
        design_vars_names = list(design_vars)
        self.num_design_vars = len(design_vars_names)

        # Generate an LHC of the proper size
        rand_lhc = self._get_lhc()

        # Map LHC to buckets
        buckets = dict()
        for j in range(self.num_design_vars):
            bounds = design_vars[design_vars_names[j]]
            design_var_buckets = self._get_buckets(bounds['low'], bounds['high'])
            buckets[design_vars_names[j]] = list()
            for i in range(self.num_samples):
                buckets[design_vars_names[j]].append(design_var_buckets[rand_lhc[i,j]])

        # Return random values in given buckets
        for i in moves.xrange(self.num_samples):
            yield dict(((key, np.random.uniform(bounds[i][0], bounds[i][1])) for key, bounds in iteritems(buckets)))

    def _get_lhc(self):
        return rand_latin_hypercube(self.num_samples, self.num_design_vars).astype(int)

    def _get_buckets(self, low, high):
        bucket_walls = np.linspace(low, high, self.num_samples + 1)
        return list(moves.zip(bucket_walls[0:-1], bucket_walls[1:]))

'''
class OptimizedLatinHypercubeDriver(LatinHypercubeDriver):
    def __init__(self, num_samples=None, population=None, generations=None):
        super(OptimizedLatinHypercubeDriver, self, numsamples).__init__()
        self.qs = [1,2,5,10,20,50,100] #list of qs to try for Phi_q optimization
        self.population = population
        self.generations = generations

    def _get_lhc(self):
        rand_lhc = rand_latin_hypercube(self.num_samples, self.)
        # Optimize our LHC before returning it
'''

'''
class LHC_indivudal(object):

    def __init__(self, doe, q=2, p=1):
        self.q = q
        self.p = p
        self.doe = doe
        self.phi = None # Morris-Mitchell sampling criterion

    @property
    def shape(self):
        """Size of the LatinHypercube DOE (rows,cols)."""
        return self.doe.shape

    def mmphi(self):
        """Returns the Morris-Mitchell sampling criterion for this Latin hypercube."""

        if self.phi is None:
            n,m = self.doe.shape
            distdict = {}

            #calculate the norm between each pair of points in the DOE
            # TODO: This norm takes up the majority of the computation time. It
            # should be converted to C or ShedSkin.
            arr = self.doe
            for i in range(n):
                for j in range(i+1, n):
                    nrm = norm(arr[i]-arr[j], ord=self.p)
                    distdict[nrm] = distdict.get(nrm, 0) + 1

            distinct_d = array(distdict.keys())

            #mutltiplicity array with a count of how many pairs of points have a given distance
            J = array(distdict.values())

            self.phi = sum(J*(distinct_d**(-self.q)))**(1.0/self.q)

        return self.phi

    def perturb(self, mutation_count):
        """ Interchanges pairs of randomly chosen elements within randomly chosen
        columns of a DOE a number of times. The result of this operation will also
        be a Latin hypercube.
        """
        new_doe = self.doe.copy()
        n,k = self.doe.shape
        for count in range(mutation_count):
            col = randint(0, k-1)

            #choosing two distinct random points
            el1 = randint(0, n-1)
            el2 = randint(0, n-1)
            while el1==el2:
                el2 = randint(0, n-1)

            new_doe[el1, col] = self.doe[el2, col]
            new_doe[el2, col] = self.doe[el1, col]

        return LHC_indivudal(new_doe, self.q, self.p)

    def __iter__(self):
        return self._get_rows()

    def _get_rows(self):
        for row in self.doe:
            yield row

    def __repr__(self):
        return repr(self.doe)

    def __str__(self):
        return str(self.doe)

    def __getitem__(self,*args):
        return self.doe.__getitem__(*args)


_norm_map = {"1-norm":1,"2-norm":2}

'''