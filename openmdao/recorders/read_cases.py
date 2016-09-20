from openmdao.recorders.sqlite_reader import SqliteCaseReader
from openmdao.recorders.hdf5_reader import HDF5CaseReader

def read_cases(filename):

    try:
        reader = SqliteCaseReader(filename)
        return reader
    except ValueError:
        pass

    try:
        reader = HDF5CaseReader(filename)
        return reader
    except:
        raise ValueError('Unable to load cases from file {0}'.format(filename))
