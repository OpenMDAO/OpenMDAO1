"""Class definition for CsvRecorder, a recorder that saves the output into a csv file."""

import csv
import numpy
import sys

from six import string_types

from openmdao.recorders.base_recorder import BaseRecorder

def serialize(val):
    """ Turn every piece of data into a string; arrays are comma
    separated.

    Args
    -----
    val : object
        Object to serialize

    Returns
    -------
    object : serialized object
    """
    if isinstance(val, numpy.ndarray):
        return ",".join(map(str, val))
    return str(val)


class CsvRecorder(BaseRecorder):
    """ Recorder that saves cases into a CSV file. This recorder does not
    record metadata.

    Args
    ----
    out : stream
        Output stream or file name to write the csv file.

    Options
    -------
    options['record_unknowns'] :  bool(True)
        Tells recorder whether to record the unknowns vector.
    options['record_params'] :  bool(False)
        Tells recorder whether to record the params vector.
    options['record_resids'] :  bool(False)
        Tells recorder whether to record the ressiduals vector.
    options['record_derivs'] :  bool(True)
        Tells recorder whether to record derivatives that are requested by a `Driver`.
    options['includes'] :  list of strings
        Patterns for variables to include in recording.
    options['excludes'] :  list of strings
        Patterns for variables to exclude in recording (processed after includes).
    """

    def __init__(self, out=sys.stdout):
        super(CsvRecorder, self).__init__()

        self.options['record_metadata'] = False
        self._wrote_header = False
        self._parallel = False
        self.ncol = 0

        if out != sys.stdout:
            # filename or file descriptor
            if isinstance(out, string_types):
                # filename was given
                out = open(out, 'w')
            self.out = out
        self.writer = csv.writer(out)

    def record_metadata(self, group):
        """Currently not supported for csv files. Do nothing.

        Args
        ----
        group : `System`
            `System` containing vectors
        """
        # TODO: what to do here?
        pass

    def record_iteration(self, params, unknowns, resids, metadata):
        """Record the current iteration.

        Args
        ----
        params : `VecWrapper`
            `VecWrapper` containing parameters. (p)

        unknowns : `VecWrapper`
            `VecWrapper` containing outputs and states. (u)

        resids : `VecWrapper`
            `VecWrapper` containing residuals. (r)

        metadata : dict
            Dictionary containing execution metadata (e.g. iteration coordinate).
        """

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
            if self.options['record_derivs']:
                header.append('Derivatives')
            self.ncol = len(header)
            self.writer.writerow(header)
            self._wrote_header = True

        row = []
        if self.options['record_params']:
            row.extend((serialize(value) for value in params.values()))
        if self.options['record_unknowns']:
            row.extend((serialize(value) for value in unknowns.values()))
        if self.options['record_resids']:
            row.extend((serialize(value) for value in resids.values()))
        if self.options['record_derivs']:
            row.append(None)
        self.writer.writerow(row)

        if self.out:
            self.out.flush()

    def record_derivatives(self, derivs, metadata):
        """Writes the derivatives that were calculated for the driver.

        Args
        ----
        derivs : dict
            Dictionary containing derivatives

        metadata : dict, optional
            Dictionary containing execution metadata (e.g. iteration coordinate).
        """

        row = [None]*(self.ncol-1)
        row.append(str([derivs]))
        self.writer.writerow(row)

        if self.out:
            self.out.flush()
