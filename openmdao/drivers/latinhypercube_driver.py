"""
OpenMDAO design-of-experiments driver implementing the Latin Hypercube method.
"""

from openmdao.drivers.predeterminedruns_driver import PredeterminedRunsDriver
import six.moves
import numpy

class LatinHypercubeDriver(PredeterminedRunsDriver):
    def __init__(self, *args, **kwargs):
        super(LatinHypercubeDriver, self).__init__(*args, **kwargs)

    def _build_runlist(self):
        design_vars = self.get_desvar_metadata()
        design_vars_names = list(design_vars)
        buckets = dict()
        for design_var_name in design_vars_names:
            bounds = design_vars[design_var_name]
            bucket_walls = numpy.linspace(bounds['low'], bounds['high'], num=self.num_steps + 1)
            buckets[design_var_name] = list(six.moves.zip(bucket_walls[0:-1], bucket_walls[1:]))
            numpy.random.shuffle(buckets[design_var_name])

        for i in six.moves.xrange(self.num_steps):
            yield dict(((key, numpy.random.uniform(bounds[i][0], bounds[i][1])) for key, bounds in buckets.iteritems()))
