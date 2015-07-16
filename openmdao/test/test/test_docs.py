import unittest
import os
import sys
import tempfile
import shutil
from subprocess import Popen, PIPE, STDOUT

def _run_subproc_test(tname):
    dn = os.path.dirname
    docdir = os.path.join(dn(dn(dn(__file__))), 'docs')

    p = Popen(['make', tname], stdout=PIPE, stderr=STDOUT,
              env=os.environ, cwd=docdir)
    return p.communicate()[0], p.returncode

class SphinxDocsTestCase(unittest.TestCase):
    def test_docs(self):
        # can't put these in separate test methods because they won't work
        # when run concurrently
        output, retcode = _run_subproc_test('html')
        if retcode:
            self.fail('problem building html sphinx docs:\\n'+output)

        # check for build warnings
        for line in output.split('\n'):
            if 'build succeeded,' in line:
                self.fail('warning while building html sphinx docs:\\n'+output)

        output, retcode = _run_subproc_test('doctest')
        if retcode:
            self.fail('problem doc testing sphinx docs:\\n'+output)

        output, retcode = _run_subproc_test('linkcheck')
        if retcode:
            self.fail('problem link checking sphinx docs:\\n'+output)


if __name__ == '__main__':
    unittest.main()
