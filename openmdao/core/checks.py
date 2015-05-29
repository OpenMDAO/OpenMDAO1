
from six.moves import zip

class ConnectError(Exception):
    @classmethod
    def type_mismatch_error(cls, src, target):
        msg = "Type '{src[type]}' of source '{src[relative_name]}' must be the same as type '{target[type]}' of target '{target[relative_name]}'"
        msg = msg.format(src=src, target=target)

        return cls(msg)

    @classmethod
    def shape_mismatch_error(cls, src, target):
        msg  = "Shape '{src[shape]}' of the source '{src[relative_name]}' must match the shape '{target[shape]}' of the target '{target[relative_name]}'"
        msg = msg.format(src=src, target=target)

        return cls(msg)

def __make_metadata(metadata):
    '''
    Add type field to metadata dict.
    Returns a modified copy of `metadata`.
    '''
    metadata = dict(metadata)
    metadata['type'] = type(metadata['val'])

    return metadata

def __get_metadata(paths, metadata_dict):
    metadata = []

    for path in paths:
        var_metadata = metadata_dict[path]
        metadata.append(__make_metadata(var_metadata))

    return metadata

def check_types_match(src, target):
    if src['type'] != target['type']:
        raise ConnectError.type_mismatch_error(src, target)

def check_connections(connections, params, unknowns):
    # Get metadata for all sources
    sources = __get_metadata(connections.values(), unknowns)

    #Get metadata for all targets
    targets = __get_metadata(connections.keys(), params)

    for source, target in zip(sources, targets):
        check_types_match(source, target)
        check_shapes_match(source, target)

def check_shapes_match(source, target):
    if source['shape'] != target['shape']:
        raise ConnectError.shape_mismatch_error(source, target)
