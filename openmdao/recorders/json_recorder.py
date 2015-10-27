""" Class definition for JsonRecorder, a recorder that
saves the output into a json file."""

import json
import numpy
import sys
import os
import inspect

from six import string_types, iteritems

from openmdao.recorders.base_recorder import BaseRecorder
from openmdao.util.record_util import format_iteration_coordinate

class JsonRecorder(BaseRecorder):

    def __init__(self, out=sys.stdout):
        super(JsonRecorder, self).__init__()

        self._first_entry = True
        self._parallel = False

        if out != sys.stdout:
            # filename or file descriptor
            if isinstance(out, string_types):
                # filename was given
                out = open(out, 'w')
        self.out = out
        out.write('{' + os.linesep + '"iterations": [' + os.linesep)

    def startup(self, group):
        super(JsonRecorder, self).startup(group)

    def record_iteration(self, params, unknowns, resids, metadata):
        # if self._wrote_header is False:
        #     self.writer.writerow([param for param in params] + [unknown for unknown in unknowns])
        #     self._wrote_header = True

        # def munge(val):
        #     if isinstance(val, numpy.ndarray):
        #         return ",".join(map(str, val))
        #     return str(val)
        # self.writer.writerow([munge(value['val']) for value in params.values()] + [munge(value['val']) for value in unknowns.values()])
        if self._first_entry:
            self._first_entry = False
        else:
            self.out.write("," + os.linesep)

        self.out.write(self.make_iteration_json(params, unknowns, resids, metadata))

        if self.out:
            self.out.flush()

    def make_iteration_json(self, params, unknowns, resids, metadata):
        iteration_dict = {}

        iteration_coordinate = metadata['coord']
        timestamp = metadata['timestamp']
        params, unknowns, resids = self._filter_vectors(params, unknowns, resids, iteration_coordinate)

        iteration_dict["coord"] = iteration_coordinate
        iteration_dict["timestamp"] = timestamp

        if self.options['record_params']:
            iteration_dict["params"] = {}
            params_dict = iteration_dict["params"]
            for param, val in sorted(iteritems(params)):
                params_dict[param] = val

        if self.options['record_unknowns']:
            iteration_dict["unknowns"] = {}
            unknown_dict = iteration_dict["unknowns"]
            for unknown, val in sorted(iteritems(unknowns)):
                unknown_dict[unknown] = val

        if self.options['record_resids']:
            iteration_dict["resids"] = {}
            resid_dict = iteration_dict["resids"]
            for resid, val in sorted(iteritems(resids)):
                resid_dict[resid] = val

        return json.dumps(iteration_dict)

    def record_metadata(self, group):
        pass
        # TODO: what to do here?
        # self.writer.writerow([param.name for param in group.params] + [unknown.name for unknowns in group.unknowns])

    def close(self):
        self.out.write(os.linesep + "]" + os.linesep + "}" + os.linesep)
        super(JsonRecorder, self).close()
