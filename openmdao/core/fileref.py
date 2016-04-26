"""
Support for file variables.
"""

import sys
import copy
import os
import shutil
from six import iteritems

#Public Symbols
__all__ = ['FileRef']

_file_meta = {
    'binary': bool,
}

class FileRef(object):
    """
    A reference to a file on disk. As well as containing metadata information,
    it supports :meth:`open` to read and write the file's contents.
    """

    def __init__(self, fname=None, parent_dir=None):
        self.fname = fname
        self.parent_dir = parent_dir
        self.meta = {}

    def __str__(self):
        return "FileRef(%s): absolute: %s" % (self.fname, self._abspath())

    def _set_meta(self, meta):
        for name, typ in iteritems(_file_meta):
            if name in meta:
                self.meta[name] = typ(meta[name])

    def open(self, mode):
        """ Open file for reading or writing. """
        if self.meta.get('binary') and 'b' not in mode:
            mode += 'b'
        return open(self._abspath(), mode)

    def _abspath(self):
        """ Return absolute path to file. """
        if os.path.isabs(self.fname):
            return self.fname
        else:
            return os.path.join(self.parent_dir, self.fname)

    def validate(self, src_fref):
        """
        validate() is called on a target `FileRef` to ensure that the source
        is a `FileRef` and that it has matching metadata.  Currently, the only
        metadata is a binary flag. Other metadata may be added in the future.
        If the metadata does not match, an exception will be raised.

        Args
        ----
        src_fref : `FileRef`
            Source `FileRef` object.
        """
        if not isinstance(src_fref, FileRef):
            raise TypeError("Source for FileRef '%s' is not a FileRef." %
                             self.fname)
        for name, typ in iteritems(_file_meta):
            if name in self.meta or name in src_fref.meta:
                tgtval = typ(self.meta.get(name))
                srcval = typ(src_fref.meta.get(name))
                if tgtval != srcval:
                    raise ValueError("Source FileRef has (%s=%s) and dest has (%s=%s)."%
                                     (name, srcval, name, tgtval))

    def _same_file(self, fref):
        """Returns True if this FileRef and the given FileRef refer to the
        same file.
        """
        # TODO: check here if we're on the same host
        return self._abspath() == fref._abspath()

    def _assign_to(self, src_fref):
        """Called by the framework during data passing when a target FileRef
        is connected to a source FileRef.  Validation is performed and the
        source file will be copied over to the destination path if it differs
        from the path of the source.
        """
        self.validate(src_fref)

        # If we refer to the same file as the source, do nothing
        if self._same_file(src_fref):
            return

        with src_fref.open("r") as src, self.open("w") as dst:
            shutil.copyfileobj(src, dst)
