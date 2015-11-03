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

        self.jsonToWrite = {}

    def startup(self, group):
        super(JsonRecorder, self).startup(group)

    def record_iteration(self, params, unknowns, resids, metadata):
        if not "iterations" in self.jsonToWrite:
            self.jsonToWrite["iterations"] = []

        self.jsonToWrite["iterations"].append(self.make_iteration_json(params, unknowns, resids, metadata))

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

        # Do a deep copy of our results dict:  there's no guarantee that openmdao isn't going
        # to try and reuse one of the objects that we're trying to store until the end of the
        # run when we write JSON (and, in fact, it does reuse the list at metadata['coord'],
        # resulting in incorrect coordinates for all but the last iteration if we don't do a
        # deep copy here).
        return copy.deepcopy(iteration_dict)

    def record_metadata(self, group):
        params = list(iteritems(group.params))
        unknowns = list(iteritems(group.unknowns))
        resids = list(iteritems(group.resids))

        self.jsonToWrite["metadata"] = {
            "params": params,
            "unknowns": unknowns,
            "resids": resids
        }

    def close(self):
        self.out.write(json.dumps(self.jsonToWrite, default=_json_encode_fallback))
        super(JsonRecorder, self).close()

def _json_encode_fallback(object):
    if isinstance(object, array):
        return object.tolist()
    elif isinstance(object, numpy.ndarray):
        return object.tolist()
    else:
        raise TypeError("No fallback encoding for type {0}".format(type(object).__name__))
