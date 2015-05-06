from collections import namedtuple

class ConnectError(Exception):
    pass

def make_var_metadata(path, metadata):
    return VarMetadata(path, metadata['val'], type(metadata['val']), metadata.get('shape'))

def check_types_match(src, target):
    if src.type != target.type:
        msg = "Type '{src.path} of source '{src.type}' must be the same as type '{target.path}' of target '{target.type}'"
        msg = msg.format(src=src, target=target)

        raise ConnectError(msg)

def check_connections(connections, params, unknowns):
    # Get metadata for all sources
    sources = map(make_var_metadata, connections.values(), map(unknowns.get, connections.values()))
    
    #Get metadata for all targets
    targets = map(make_var_metadata, connections.keys(), map(params.get, connections.keys()))
    
    for source, target in zip(sources, targets):
        check_types_match(source, target)
        check_shapes_match(source, target)

def check_shapes_match(source, target):
    #Use the type of the shape of source and target to determine which the #correct function to use for shape checking
    source_shape_type = type(source.shape)
    target_shape_type = type(target.shape)
    
    check_shape_function = __shape_checks.get((source_shape_type, target_shape_type), lambda x, y: None)
    
    check_shape_function(source, target)

def __check_shapes_match(src, target):
    if src.shape != target.shape:
        msg  = "Shape '{src.shape}' of the source '{src.path}' must match the shape '{target.shape}' of the target '{target.path}'"
        msg = msg.format(src=src, target=target)

        raise ConnectError(msg)

def __check_val_and_shape_match(src, target):
    if src.val.shape != target.shape:
        msg = "Shape of the initial value '{src.val.shape}' of source '{src.path}' must match the shape '{target.shape}' of the target '{target.path}'"
        msg = msg.format(src=src, target=target)

        raise ConnectError(msg)

VarMetadata = namedtuple('VarMetadata', 'path val type shape')
        
__shape_checks = {
    (tuple, tuple) : __check_shapes_match,
    (type(None), tuple)  : __check_val_and_shape_match
}
