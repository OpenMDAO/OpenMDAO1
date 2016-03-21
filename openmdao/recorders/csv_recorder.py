"""Class definition for CsvRecorder, a recorder that saves the output into a csv file."""

import csv
import numpy
import sys

from six import string_types, itervalues

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
        if self.options['record_metadata']:
            raise RuntimeError("Recording of metadata is not supported by CsvRecorder.")

    def record_iteration(self, params, unknowns, resids, metadata):
        """Record the current iteration. The first column will always be
        a variable called 'success', which will have a value of 1 if iteration
        was successful and 0 if not.

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

        if self.options['record_params']:
            params = self._filter_vector(params, 'p', iteration_coordinate)
        else:
            params = None
        if self.options['record_unknowns']:
            unknowns = self._filter_vector(unknowns, 'u', iteration_coordinate)
        else:
            unknowns = None
        if self.options['record_resids']:
            resids = self._filter_vector(resids, 'r', iteration_coordinate)
        else:
            resids = None

        if self._wrote_header is False:
            header = ['success'] # add column for success flag
            if params is not None:
                header.extend(params)
            if unknowns is not None:
                header.extend(unknowns)
            if resids is not None:
                header.extend(resids)
            if self.options['record_derivs']:
                header.append('Derivatives')

            self.ncol = len(header)
            self.writer.writerow(header)
            self._wrote_header = True

        row = [metadata['success']] # add column for success flag

        if params is not None:
            row.extend(serialize(value) for value in itervalues(params))
        if unknowns is not None:
            row.extend(serialize(value) for value in itervalues(unknowns))
        if resids is not None:
            row.extend(serialize(value) for value in itervalues(resids))
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

        # put None's in all of the non-derivative columns except
        # the success column
        row = [None]*(self.ncol-1)
        row[0] = metadata['success']

        row.append(str([derivs]))
        self.writer.writerow(row)

        if self.out:
            self.out.flush()
