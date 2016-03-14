""" Set of utilities for detecting and reporting connection errors."""

import traceback
from six import iteritems
from openmdao.core.fileref import FileRef

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
    errors = []

    for tgt, (src, idxs) in iteritems(connections):
        tmeta = params_dict[tgt]
        smeta = unknowns_dict[src]
        _check_types_match(smeta, tmeta, to_prom_name)
        if 'pass_by_obj' not in smeta and 'pass_by_obj' not in smeta:
            errors.extend(_check_shapes_match(smeta, tmeta, to_prom_name))

    # FileRefs require a separate check, because if multiple input FileRefs
    # refer to the same file, that means they're implicitly connected,
    # even if they're not explicitly connected or connected by promotion in
    # the framework. If multiple input FileRefs refer to the same file, they
    # must all share the same source FileRef or we raise an exception.
    fref_conns = {}  # dict of input varname vs ([(invar,connected)...], [(outvar, outfile)...])
    for n, meta in iteritems(params_dict):
        val = meta['val']
        if isinstance(val, FileRef):
            ins, outs = fref_conns.setdefault(val._abspath(), ([], set()))
            if n in connections:
                ins.append((n, True))
                s,_ = connections[n]
                outs.add((s, unknowns_dict[s]['val']._abspath()))
            else:
                ins.append((n, False))

    for infile, (ins, outs) in iteritems(fref_conns):
        if len(outs) > 1:
            errors.append("Input file '%s' is referenced from FileRef param(s) %s, "
                          "which are connected to multiple "
                          "output FileRefs: %s. Those FileRefs reference the following "
                          "files: %s." % (infile, [i for i,isconn in ins],
                                          sorted([o for o,of in outs]),
                                          sorted([of for o,of in outs])))

        for ivar, isconn in ins:
            if not isconn and outs:
                errors.append("FileRef param '%s' is unconnected but will be "
                              "overwritten by the following FileRef unknown(s): "
                              "%s. Files referred to by the FileRef unknowns are: "
                              "%s. To remove this error, make a connection between %s"
                              " and a FileRef unknown." % (ivar,
                                          sorted([o for o,of in outs]),
                                          sorted([of for o,of in outs]), ivar))

    return errors

def _check_types_match(src, tgt, to_prom_name):
    sval = src['val']
    tval = tgt['val']

    if isinstance(sval, FileRef) or isinstance(tval, FileRef):
        tval.validate(sval)
        return

    stype = type(sval)
    ttype = type(tval)

    if stype == ttype:
        return

    if 'src_indices' in tgt:
        if len(tgt['src_indices']) == 1 and ttype == float:
            return

    raise TypeError("Type %s of source %s must be the same as type %s of "
                    "target %s." %  (type(src['val']),
                    _both_names(src, to_prom_name),
                    type(tval), _both_names(tgt, to_prom_name)))

def _check_shapes_match(source, target, to_prom_name):
    sshape = source.get('shape')
    tshape = target.get('shape')
    if sshape is not None and tshape is not None:
        return __check_shapes_match(source, target, to_prom_name)
    elif sshape is None and tshape is not None:
        return __check_val_and_shape_match(source, target, to_prom_name)
    return ()

def __check_shapes_match(src, target, to_prom_name):
    errors = []

    src_idxs = target.get('src_indices')

    if src['shape'] != target['shape']:
        if src_idxs is not None:
            if len(src_idxs) != target['size']:
                errors.append("Size %d of the indexed sub-part of source "
                                 "%s must be the same as size %d of target "
                                 "%s." %
                                 (len(target['src_indices']),
                                 _both_names(src, to_prom_name),
                                 target['size'],
                                 _both_names(target, to_prom_name)))
        elif 'src_indices' in src:
            if target['size'] != src['distrib_size']:
                errors.append("Total size %d of  distributed source "
                        "%s must be the same as size "
                        "%d of target %s." %
                        (src['distrib_size'], _both_names(src, to_prom_name),
                         target['size'], _both_names(target, to_prom_name)))
        else:
            errors.append("Shape %s of source %s must be the same as shape "
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
            errors.append("%s src_indices contains an index (%d) that "
                             "exceeds the bounds of source variable %s of "
                             "size %d." %
                             (_both_names(target, to_prom_name),
                             max_idx, _both_names(src, to_prom_name), ssize))
        min_idx = min(src_idxs)
        if min_idx < 0:
            errors.append("%s src_indices contains a negative index "
                             "(%d)." %
                             (_both_names(target, to_prom_name), min_idx))

    return errors

def __check_val_and_shape_match(src, target, to_prom_name):
    if src['val'].shape != target['shape']:
        return ["Shape of the initial value %s of source "
                         "%s must be the same as shape %s of target %s." %
                         (src[val].shape, _both_names(src, to_prom_name),
                         target['shape'], _both_names(target, to_prom_name))]
    return ()

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
