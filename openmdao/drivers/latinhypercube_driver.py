"""
OpenMDAO design-of-experiments Driver implementing the Latin Hypercube and Optimized Latin Hypercube methods.
"""

from openmdao.drivers.predeterminedruns_driver import PredeterminedRunsDriver
from six import moves, iteritems, itervalues
from random import shuffle, randint, seed
import numpy as np


class LatinHypercubeDriver(PredeterminedRunsDriver):
    """Design-of-experiments Driver implementing the Latin Hypercube method.
    """

    def __init__(self, num_samples=1, seed=None):
        super(LatinHypercubeDriver, self).__init__()
        self.num_samples = num_samples
        self.seed = seed

    def _build_runlist(self):
        """Build a runlist based on the Latin Hypercube method."""
        design_vars = self.get_desvar_metadata()
        design_vars_names = list(design_vars)
        self.num_design_vars = len(design_vars_names)
        if self.seed is not None:
            seed(self.seed)
            np.random.seed(self.seed)

        # Generate an LHC of the proper size
        rand_lhc = self._get_lhc()

        # Map LHC to buckets
        buckets = dict()
        for j in range(self.num_design_vars):
            bounds = design_vars[design_vars_names[j]]
            design_var_buckets = self._get_buckets(bounds['lower'], bounds['upper'])
            buckets[design_vars_names[j]] = list()
            for i in range(self.num_samples):
                buckets[design_vars_names[j]].append(design_var_buckets[rand_lhc[i, j]])

        # Return random values in given buckets
        for i in moves.xrange(self.num_samples):
            yield dict(((key, np.random.uniform(bounds[i][0], bounds[i][1])) for key, bounds in iteritems(buckets)))

    def _get_lhc(self):
        """Generates a Latin Hypercube based on the number of samples and the number of design variables."""

        rand_lhc = _rand_latin_hypercube(self.num_samples, self.num_design_vars)
        return rand_lhc.astype(int)

    def _get_buckets(self, low, high):
        """Determines the distribution of samples."""

        bucket_walls = np.linspace(low, high, self.num_samples + 1)
        return list(moves.zip(bucket_walls[0:-1], bucket_walls[1:]))


class OptimizedLatinHypercubeDriver(LatinHypercubeDriver):
    """Design-of-experiments Driver implementing the Morris-Mitchell method for an Optimized Latin Hypercube.
    """

    def __init__(self, num_samples=1, seed=None, population=20, generations=2, norm_method=1):
        super(OptimizedLatinHypercubeDriver, self).__init__()
        self.qs = [1, 2, 5, 10, 20, 50, 100]  # List of qs to try for Phi_q optimization
        self.num_samples = num_samples
        self.seed = seed
        self.population = population
        self.generations = generations
        self.norm_method = norm_method

    def _get_lhc(self):
        """Generate an Optimized Latin Hypercube
        """

        rand_lhc = _rand_latin_hypercube(self.num_samples, self.num_design_vars)

        # Optimize our LHC before returning it
        best_lhc = _LHC_Individual(rand_lhc, q=1, p=self.norm_method)
        for q in self.qs:
            lhc_start = _LHC_Individual(rand_lhc, q, self.norm_method)
            lhc_opt = _mmlhs(lhc_start, self.population, self.generations)
            if lhc_opt.mmphi() < best_lhc.mmphi():
                best_lhc = lhc_opt

        return best_lhc._get_doe().astype(int)


class _LHC_Individual(object):
    def __init__(self, doe, q=2, p=1):
        self.q = q
        self.p = p
        self.doe = doe
        self.phi = None  # Morris-Mitchell sampling criterion

    @property
    def shape(self):
        """Size of the LatinHypercube DOE (rows,cols)."""

        return self.doe.shape

    def mmphi(self):
        """Returns the Morris-Mitchell sampling criterion for this Latin hypercube."""

        if self.phi is None:
            n, m = self.doe.shape
            distdict = {}

            # Calculate the norm between each pair of points in the DOE
            arr = self.doe
            for i in range(1, n):
                nrm = np.linalg.norm(arr[i] - arr[:i], ord=self.p, axis=1)
                for j in range(0, i):
                    nrmj = nrm[j]
                    if nrmj in distdict:
                        distdict[nrmj] += 1
                    else:
                        distdict[nrmj] = 1

            distinct_d = np.array(list(distdict))

            # Mutltiplicity array with a count of how many pairs of points have a given distance
            J = np.array(list(itervalues(distdict)))

            self.phi = sum(J * (distinct_d ** (-self.q))) ** (1.0 / self.q)

        return self.phi

    def perturb(self, mutation_count):
        """ Interchanges pairs of randomly chosen elements within randomly chosen
        columns of a DOE a number of times. The result of this operation will also
        be a Latin hypercube.
        """

        new_doe = self.doe.copy()
        n, k = self.doe.shape
        for count in range(mutation_count):
            col = randint(0, k - 1)

            # Choosing two distinct random points
            el1 = randint(0, n - 1)
            el2 = randint(0, n - 1)
            while el1 == el2:
                el2 = randint(0, n - 1)

            new_doe[el1, col] = self.doe[el2, col]
            new_doe[el2, col] = self.doe[el1, col]

        return _LHC_Individual(new_doe, self.q, self.p)

    def __iter__(self):
        return self._get_rows()

    def _get_rows(self):
        for row in self.doe:
            yield row

    def __repr__(self):
        return repr(self.doe)

    def __str__(self):
        return str(self.doe)

    def __getitem__(self, *args):
        return self.doe.__getitem__(*args)

    def _get_doe(self):
        return self.doe


def _rand_latin_hypercube(n, k):
    # Calculates a random Latin hypercube set of n points in k dimensions within [0,n-1]^k hypercube.
    arr = np.zeros((n, k))
    row = list(moves.xrange(0, n))
    for i in moves.xrange(k):
        shuffle(row)
        arr[:, i] = row
    return arr


def _is_latin_hypercube(lh):
    """Returns True if the given array is a Latin hypercube.
    The given array is assumed to be a numpy array.
    """

    n, k = lh.shape
    for j in range(k):
        col = lh[:, j]
        colset = set(col)
        if len(colset) < len(col):
            return False  # something was duplicated
    return True


def _mmlhs(x_start, population, generations):
    """Evolutionary search for most space filling Latin-Hypercube.
    Returns a new LatinHypercube instance with an optimized set of points.
    """

    x_best = x_start
    phi_best = x_start.mmphi()
    n = x_start.shape[1]

    level_off = np.floor(0.85 * generations)
    for it in range(generations):
        if it < level_off and level_off > 1.:
            mutations = int(round(1 + (0.5 * n - 1) * (level_off - it) / (level_off - 1)))
        else:
            mutations = 1

        x_improved = x_best
        phi_improved = phi_best

        for offspring in range(population):
            x_try = x_best.perturb(mutations)
            phi_try = x_try.mmphi()

            if phi_try < phi_improved:
                x_improved = x_try
                phi_improved = phi_try

        if phi_improved < phi_best:
            phi_best = phi_improved
            x_best = x_improved

    return x_best
