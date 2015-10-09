from openmdao.core.options import OptionsDictionary

def record_metadata(self, group):
    raise NotImplementedError("record_metadata is not implemented")

def record_iteration(self, params, unknowns, resids, metadata):
    raise NotImplementedError("record_iteration is not implemented")

metadata_options = OptionsDictionary()
metadata_options.add_option('record_metadata', True)

iteration_options = OptionsDictionary()
iteration_options.add_option('record_unknowns', True)
iteration_options.add_option('record_params', False)
iteration_options.add_option('record_resids', False)
iteration_options.add_option('includes', ['*'])
iteration_options.add_option('excludes', [])

class MetadataRecorder(object):
    options = metadata_options
    methods = [record_metadata]

class IterationRecorder(object):
    options = iteration_options
    methods = [record_iteration]



