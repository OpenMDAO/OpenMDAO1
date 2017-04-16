""" Class definition for JsonRecorder, a recorder that
saves the output into a json file."""

import json
import numpy
import sys
import os
import copy

from array import array
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

        self.out.write("{")

    def startup(self, group):
        super(JsonRecorder, self).startup(group)

    def record_iteration(self, params, unknowns, resids, metadata):
        if self._first_entry:
            self.out.write('"iterations": [' + os.linesep)
            self._first_entry = False
        else:
            self.out.write("," + os.linesep)

        iterationObject = self.make_iteration_json(params, unknowns, resids, metadata)
        self.out.write(json.dumps(iterationObject, indent=2, default=_json_encode_fallback))

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

        return iteration_dict

    def record_metadata(self, group):
        params = list(iteritems(group.params))
        unknowns = list(iteritems(group.unknowns))
        resids = list(iteritems(group.resids))

        metadataObject = {
            "params": params,
            "unknowns": unknowns,
            "resids": resids
        }

        # Write the metadata to the output stream.
        # Assumes that it's alway called before the first call to record_iteration
        self.out.write('"metadata": ' + json.dumps(metadataObject, indent=2, default=_json_encode_fallback) + "," + os.linesep)
        self.out.flush()

    def close(self):
        # Close out the JSON object to make it nice and well-formed
        if self._first_entry:
            # Write empty iterations object if, for some reason, record_iteration were never called
            self.out.write('"iterations": [' + os.linesep)
            self._first_entry = False

        self.out.write(']' + os.linesep + "}")

        self.out.write(os.linesep) # trailing newline
        super(JsonRecorder, self).close()

def _json_encode_fallback(object):
    if isinstance(object, array):
        return object.tolist()
    elif isinstance(object, numpy.ndarray):
        return object.tolist()
    else:
        raise TypeError("No fallback encoding for type {0}".format(type(object).__name__))
