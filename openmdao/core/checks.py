
from six.moves import zip
from openmdao.units.units import PhysicalQuantity

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

    @classmethod
    def val_and_shape_mismatch_error(cls, src, target):
        msg = "Shape of the initial value '{src[val].shape}' of source '{src[relative_name]}' must match the shape '{target[shape]}' of the target '{target[relative_name]}'"
        msg = msg.format(src=src, target=target)

        return cls(msg)

    @classmethod
    def unit_mismatch_error(cls, src, target):
        msg = "Unit '{src[val].unit}' of the source '{src[relative_name]}' must be compatible with the unit '{target[val].unit}' of the target '{target[relative_name]}'"
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

def check_units_match(src, target):
    if isinstance(src['val'], PhysicalQuantity) and isinstance(target['val'], PhysicalQuantity):
        if not src['val'].is_compatible(target['val'].unit):
            raise ConnectError.unit_mismatch_error(src, target)

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
        check_units_match(source, target)

def check_shapes_match(source, target):
    #Use the type of the shape of source and target to determine which the #correct function to use for shape checking

    check_shape_function = __shape_checks.get((type(source.get('shape')), type(target.get('shape'))), lambda x, y: None)

    check_shape_function(source, target)

def __check_shapes_match(src, target):
    if src['shape'] != target['shape']:
        raise ConnectError.shape_mismatch_error(src, target)

def __check_val_and_shape_match(src, target):
    if src['val'].shape != target['shape']:
        raise ConnectError.val_and_shape_mismatch_error(src, target)

__shape_checks = {
    (tuple, tuple) : __check_shapes_match,
    (type(None), tuple)  : __check_val_and_shape_match
}
