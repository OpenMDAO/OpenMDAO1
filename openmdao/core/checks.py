""" Everything here pertains to connection errors."""

from six.moves import zip

class ConnectError(Exception):
    @classmethod
    def _type_mismatch_error(cls, src, target):
        msg = "Type '{src[type]}' of source '{src[promoted_name]}' must be the same as type '{target[type]}' of target '{target[promoted_name]}'"
        msg = msg.format(src=src, target=target)

        return cls(msg)

    @classmethod
    def _shape_mismatch_error(cls, src, target):
        msg  = "Shape '{src[shape]}' of the source '{src[promoted_name]}' must match the shape '{target[shape]}' of the target '{target[promoted_name]}'"
        msg = msg.format(src=src, target=target)

        return cls(msg)

    @classmethod
    def _size_mismatch_error(cls, src, target):
        msg  = "Size {len(target[src_indices])} of the indexed sub-part of source '{src[promoted_name]}' must match the size '{target[size]}' of the target '{target[promoted_name]}'"
        msg = msg.format(src=src, target=target)

        return cls(msg)

    @classmethod
    def _indices_too_large(cls, src, target):
        msg  = "Size {len(target[src_indices])} of target indices is larger than size {src[size]} of source '{src[promoted_name]}'"
        msg = msg.format(src=src, target=target)

        return cls(msg)

    @classmethod
    def _val_and_shape_mismatch_error(cls, src, target):
        msg = "Shape of the initial value '{src[val].shape}' of source '{src[promoted_name]}' must match the shape '{target[shape]}' of the target '{target[promoted_name]}'"
        msg = msg.format(src=src, target=target)

        return cls(msg)

    @classmethod
    def nonexistent_src_error(cls, src, target):
        msg = ("Source '{src}' cannot be connected to target '{target}': "
               "'{src}' does not exist.")

        msg = msg.format(src=src, target=target)

        return cls(msg)

    @classmethod
    def nonexistent_target_error(cls, src, target):
        msg = ("Source '{src}' cannot be connected to target '{target}': "
               "'{target}' does not exist.")

        msg = msg.format(src=src, target=target)

        return cls(msg)

    @classmethod
    def invalid_target_error(cls, src, target):
        msg = ("Source '{src}' cannot be connected to target '{target}': "
               "Target must be a parameter but '{target}' is an unknown.")

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
        raise ConnectError._type_mismatch_error(src, target)

def check_connections(connections, params, unknowns):
    # Get metadata for all sources
    sources = __get_metadata(connections.values(), unknowns)

    #Get metadata for all targets
    targets = __get_metadata(connections.keys(), params)

    for source, target in zip(sources, targets):
        check_types_match(source, target)
        check_shapes_match(source, target)

def check_shapes_match(source, target):
    #Use the type of the shape of source and target to determine which the #correct function to use for shape checking

    check_shape_function = __shape_checks.get((type(source.get('shape')), type(target.get('shape'))), lambda x, y: None)

    check_shape_function(source, target)

def __check_shapes_match(src, target):
    if src['shape'] != target['shape']:
        if 'src_indices' in target:
            if len(target['src_indices']) != target['size']:
                raise ConnectError._size_mismatch_error(src, target)
            elif len(target['src_indices']) > src['size']:
                raise ConnectError._indices_too_large(src, target)
        else:
            raise ConnectError._shape_mismatch_error(src, target)

def __check_val_and_shape_match(src, target):
    if src['val'].shape != target['shape']:
        raise ConnectError._val_and_shape_mismatch_error(src, target)

__shape_checks = {
    (tuple, tuple) : __check_shapes_match,
    (type(None), tuple)  : __check_val_and_shape_match
}
