""" Misc. file utility routines. """

import itertools
import os
import warnings
import pprint
from six import string_types, iteritems

from fnmatch import fnmatch
from os.path import join, dirname, exists, abspath


def build_directory(dct, force=False, topdir='.'):
    """Create a directory structure based on the contents of a nested dict.
    The directory is created in the specified top directory or in the current
    working directory if one isn't specified. If a file being created already
    exists, a warning will be issued and the file will not be changed if force
    is False. If force is True, the file will be overwritten.

    The structure of the dict is as follows: if the value at a key is a
    dict, then that key is used to create a directory. Otherwise, the key is
    used to create a file, and the value stored at that key is written to the
    file. All keys must be relative names or a RuntimeError will be raised.

    Args
    ----
    dct : dict
        Dictionary containing nested folder structure.
    force : bool, optional
        Set to True to overwrite existing files
    topdir : string, optional
        Specify a top directory.
    """
    #TODO: if a value stored in the dict is a callable, then call it and store
    #      its return value as the contents of the file
    startdir = os.getcwd()
    topdir = os.path.abspath(topdir)
    try:
        for key, val in iteritems(dct):
            os.chdir(topdir)
            if os.path.isabs(key):
                raise RuntimeError("build_directory: key (%s) is not a relative name" % key)
            if isinstance(val, dict):  # it's a dict, so this is a directory
                if not os.path.exists(key):
                    os.makedirs(key)
                os.chdir(key)
                build_directory(val, force)
            else:  # assume a string value. Use that value to create a file
                if os.path.exists(key) and force is False:
                    warnings.warn("File '%s' already exists and will not be overwritten."
                                  % key, Warning)
                else:
                    dname = os.path.dirname(key)
                    if dname and not os.path.isdir(dname):
                        os.makedirs(dname)
                    with open(key, 'w') as f:
                        f.write(val)
    finally:
        os.chdir(startdir)


def find_files(start, match=None, exclude=None,
               dirmatch=None, direxclude=None):
    """Return filenames (using a generator).

    Walks all subdirectories below each specified starting directory,
    subject to directory filtering.

    Args
    ----
    start : str or list of str
        Starting directory or list of directories.

    match : str or predicate funct
        Either a string containing a glob pattern to match
        or a predicate function that returns True on a match.
        This is used to match files only.

    exclude : str or predicate funct
        Either a string containing a glob pattern to exclude
        or a predicate function that returns True to exclude.
        This is used to exclude files only.

    dirmatch : str or predicate funct
        Either a string containing a glob pattern to match
        or a predicate function that returns True on a match.
        This is used to match directories only.

    direxclude : str or predicate funct
        Either a string containing a glob pattern to exclude
        or a predicate function that returns True to exclude.
        This is used to exclude directories only.

    """
    startdirs = [start] if isinstance(start, string_types) else start
    if len(startdirs) == 0:
        return iter([])

    gen = _file_gen
    if match is None:
        matcher = bool
    elif isinstance(match, string_types):
        matcher = lambda name: fnmatch(name, match)
    else:
        matcher = match

    if dirmatch is None:
        dmatcher = bool
    elif isinstance(dirmatch, string_types):
        dmatcher = lambda name: fnmatch(name, dirmatch)
    else:
        dmatcher = dirmatch

    if isinstance(exclude, string_types):
        fmatch = lambda name: matcher(name) and not fnmatch(name, exclude)
    elif exclude is not None:
        fmatch = lambda name: matcher(name) and not exclude(name)
    else:
        fmatch = matcher

    if isinstance(direxclude, string_types):
        dmatch = lambda name: dmatcher(name) and not fnmatch(name, direxclude)
    elif direxclude is not None:
        dmatch = lambda name: dmatcher(name) and not direxclude(name)
    else:
        dmatch = dmatcher

    iters = [gen(d, fmatch=fmatch, dmatch=dmatch) for d in startdirs]
    if len(iters) > 1:
        return itertools.chain(*iters)
    else:
        return iters[0]


def find_up(name, path=None):
    """Search upward from the starting path (or the current directory)
    until the given file or directory is found. The given name is
    assumed to be a basename, not a path.  Returns the absolute path
    of the file or directory if found, or None otherwise.

    Args
    ----
    name : str
        Base name of the file or directory being searched for.

    path : str, optional
        Starting directory.  If not supplied, current directory is used.
    """
    if not path:
        path = os.getcwd()
    if not exists(path):
        return None
    while path:
        if exists(join(path, name)):
            return abspath(join(path, name))
        else:
            pth = path
            path = dirname(path)
            if path == pth:
                return None
    return None

def _file_gen(dname, fmatch=bool, dmatch=None):
    """A generator returning files under the given directory, with optional
    file and directory filtering.

    Args
    ----
    fmatch : predicate funct
        A predicate function that returns True on a match.
        This is used to match files only.

    dmatch : predicate funct
        A predicate function that returns True on a match.
        This is used to match directories only.
    """
    if dmatch is not None and not dmatch(dname):
        return

    for path, dirlist, filelist in os.walk(dname):
        if dmatch is not None: # prune directories to search
            newdl = [d for d in dirlist if dmatch(d)]
            if len(newdl) != len(dirlist):
                dirlist[:] = newdl # replace contents of dirlist to cause pruning

        for name in [f for f in filelist if fmatch(f)]:
            yield join(path, name)

class DirContext(object):
    """Supports using the 'with' statement in place of try-finally to
    change to and return from a directory.
    """

    def __init__(self, dpath):
        self.dpath = dpath

    def __enter__(self):
        self.start = os.getcwd()
        os.chdir(self.dpath)
        return self.dpath

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.chdir(self.start)
