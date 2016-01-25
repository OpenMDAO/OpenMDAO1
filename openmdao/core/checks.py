""" Set of utilities for detecting and reporting connection errors."""

from six import iteritems

def check_connections(connections, params_dict, unknowns_dict, to_prom_name):
    """Checks the specified connections to make sure they are valid in
    OpenMDAO.

    Args
    ----
    params_dict : dict
         A dictionary mapping absolute var name to its metadata for
         every param in the model.

    unknowns_dict : dict
         A dictionary mapping absolute var name to its metadata for
         every unknown in the model.

    to_prom_name : dict
        A dictionary mapping absolute var name to promoted var name.

    Raises
    ------
    TypeError, or ValueError
        Any invalidity in the connection raises an error.
    """
    for tgt, (src, idxs) in iteritems(connections):
        tmeta = params_dict[tgt]
        smeta = unknowns_dict[src]
        _check_types_match(smeta, tmeta, to_prom_name)
        if 'pass_by_obj' not in smeta and 'pass_by_obj' not in smeta:
            _check_shapes_match(smeta, tmeta, to_prom_name)

def _check_types_match(src, tgt, to_prom_name):
    stype = type(src['val'])
    ttype = type(tgt['val'])

    if stype == ttype:
        return

    src_indices = tgt.get('src_indices')
    if src_indices and len(src_indices) == 1 and ttype == float:
        return

    raise TypeError("Type %s of source %s must be the same as type %s of "
                    "target %s." %  (type(src['val']),
                    _both_names(src, to_prom_name),
                    type(tgt['val']), _both_names(tgt, to_prom_name)))

def _check_shapes_match(source, target, to_prom_name):
    sshape = source.get('shape')
    tshape = target.get('shape')
    if sshape is not None and tshape is not None:
        __check_shapes_match(source, target, to_prom_name)
    elif sshape is None and tshape is not None:
        __check_val_and_shape_match(source, target, to_prom_name)

def __check_shapes_match(src, target, to_prom_name):
    src_idxs = target.get('src_indices')

    if src['shape'] != target['shape']:
        if src_idxs is not None:
            if len(src_idxs) != target['size']:
                raise ValueError("Size %d of the indexed sub-part of source "
                                 "%s must be the same as size %d of target "
                                 "%s." %
                                 (len(target['src_indices']),
                                 _both_names(src, to_prom_name),
                                 target['size'],
                                 _both_names(target, to_prom_name)))
            if len(src_idxs) > src['size']:
                raise ValueError("Size %d of target indices is larger than size"
                                 " %d of source %s." %
                                 (len(src_idxs), src['size'],
                                 _both_names(src, to_prom_name)))
        elif 'src_indices' in src:
            if target['size'] != src['distrib_size']:
                raise ValueError("Total size %d of  distributed source "
                        "%s must be the same as size "
                        "%d of target %s." %
                        (src['distrib_size'], _both_names(src, to_prom_name),
                         target['size'], _both_names(target, to_prom_name)))
        else:
            raise ValueError("Shape %s of source %s must be the same as shape "
                             "%s of target %s." % (src['shape'],
                             _both_names(src, to_prom_name), target['shape'],
                             _both_names(target, to_prom_name)))

    if src_idxs is not None:
        if 'src_indices' in src:
            ssize = src['distrib_size']
        else:
            ssize = src['size']
        max_idx = max(src_idxs)
        if max_idx >= ssize:
            raise ValueError("%s src_indices contains an index (%d) that "
                             "exceeds the bounds of source variable %s of "
                             "size %d." %
                             (_both_names(target, to_prom_name),
                             max_idx, _both_names(src, to_prom_name), ssize))
        min_idx = min(src_idxs)
        if min_idx < 0:
            raise ValueError("%s src_indices contains a negative index "
                             "(%d)." %
                             (_both_names(target, to_prom_name), min_idx))

def __check_val_and_shape_match(src, target, to_prom_name):
    if src['val'].shape != target['shape']:
        raise ValueError("Shape of the initial value %s of source "
                         "%s must be the same as shape %s of target %s." %
                         (src[val].shape, _both_names(src, to_prom_name),
                         target['shape'], _both_names(target, to_prom_name)))

def _both_names(meta, to_prom_name):
    """If the pathname differs from the promoted name, return a string with
    both names. Otherwise, just return the pathname.
    """
    name = meta['pathname']
    pname = to_prom_name[name]
    if name == pname:
        return "'%s'" % name
    else:
        return "'%s' (%s)" % (name, pname)
