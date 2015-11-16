"""Class definition for CsvRecorder, a recorder that saves the output into a csv file."""

import csv
import numpy
import sys

from six import string_types

from openmdao.recorders.base_recorder import BaseRecorder


class CsvRecorder(BaseRecorder):

    def __init__(self, out=sys.stdout):
        super(CsvRecorder, self).__init__()
        
        self.options['record_metadata'] = False
        self._wrote_header = False
        self._parallel = False

        if out != sys.stdout:
            # filename or file descriptor
            if isinstance(out, string_types):
                # filename was given
                out = open(out, 'w')
            self.out = out
        self.writer = csv.writer(out)

    def startup(self, group):
        super(CsvRecorder, self).startup(group)

    def record_iteration(self, params, unknowns, resids, metadata):
        iteration_coordinate = metadata['coord']
        params, unknowns, resids = self._filter_vectors(params, unknowns, resids, iteration_coordinate)

        if self._wrote_header is False:
            header = []
            if self.options['record_params']:
                header.extend(params)
            if self.options['record_unknowns']:
                header.extend(unknowns)
            if self.options['record_resids']:
                header.extend(resids)
            self.writer.writerow(header)
            self._wrote_header = True

        def serialize(val):
            if isinstance(val, numpy.ndarray):
                return ",".join(map(str, val))
            return str(val)

        row = []
        if self.options['record_params']:
            row.extend((serialize(value) for value in params.values()))
        if self.options['record_unknowns']:
            row.extend((serialize(value) for value in unknowns.values()))
        if self.options['record_resids']:
            row.extend((serialize(value) for value in resids.values()))

        self.writer.writerow(row)

        if self.out:
            self.out.flush()

    def record_metadata(self, group):
        pass
        # TODO: what to do here?
        # self.writer.writerow([param.name for param in group.params] + [unknown.name for unknowns in group.unknowns])
