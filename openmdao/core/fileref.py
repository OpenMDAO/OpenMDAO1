"""
Support for file variables.
"""

import sys
import copy
import os

#Public Symbols
__all__ = ['FileRef']

#_big_endian = sys.byteorder == 'big'

CHUNK = 1 << 20  # 1MB

class FileRef(object):
    """
    A reference to a file on disk. As well as containing metadata information,
    it supports :meth:`open` to read and write the file's contents.
    """

    def __init__(self, fname, binary=False):
        #, desc='', content_type='', platform=sys.platform,
              #big_endian=_big_endian, single_precision=False,
              #integer_8=False, unformatted=False, recordmark_8=False):
        self.fname = fname
        self.parent_dir = None
        self.binary = binary
        # self.desc = desc
        # self.content_type = content_type
        # self.platform = platform
        # self.big_endian = big_endian
        # self.single_precision = single_precision
        # self.integer_8 = integer_8
        # self.unformatted = unformatted
        # self.recordmark_8 = recordmark_8

    def __str__(self):
        return "FileRef(%s): absolute: %s" % (self.fname, self._abspath())

    def open(self, mode):
        """ Open file for reading or writing. """
        if self.binary and 'b' not in mode:
            mode += 'b'
        return open(self._abspath(), mode)

    def _abspath(self):
        """ Return absolute path to file. """
        if os.path.isabs(self.fname):
            return self.fname
        else:
            return os.path.join(self.parent_dir, self.fname)

    def validate(self, src_fref):
        if not isinstance(src_fref, FileRef):
            raise TypeError("Source for FileRef '%s' is not a FileRef!" %
                             self.fname)
        if self.binary != src_fref.binary:
            raise ValueError("Source FileRef is (binary=%s) and dest is (binary=%s)."%
                             (src_fref.binary, self.binary))

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
            while dst.write(src.read(CHUNK)):
                pass
