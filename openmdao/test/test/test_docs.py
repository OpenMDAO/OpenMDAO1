import unittest
import os
import sys
import tempfile
import shutil
from subprocess import Popen, PIPE, STDOUT

if os.environ.get('OPENMDAO_TEST_DOCS'):
    def _run_subproc_test(tname):
        dn = os.path.dirname
        docdir = os.path.join(dn(dn(dn(os.path.abspath(__file__)))), 'docs')

        p = Popen(['make', tname], stdout=PIPE, stderr=STDOUT,
                  env=os.environ, cwd=docdir)
        return p.communicate()[0], p.returncode

    class SphinxDocsTestCase(unittest.TestCase):
        def test_docs(self):
            # we can't split these up into separate tests because the html
            # build must be complete before the others and if we split them
            # up they could be run concurrently and give erroneous results.
            output, retcode = _run_subproc_test('html')
            if retcode:
                self.fail('problem building html sphinx docs:\\n%s' % output)

            output = str(output) # fix py3 issue
            # check for build warnings
            if 'build succeeded,' in output:  # trailing comma means warnings
                self.fail('warning while building html sphinx docs:\\n%s' % output)

            output, retcode = _run_subproc_test('doctest')
            if retcode:
                self.fail('problem doc testing sphinx docs:\\n%s' % output)

            ## linkcheck seems too unreliable to run all of the time, so
            ## commenting it out for now
            # output, retcode = _run_subproc_test('linkcheck')
            # if retcode:
            #     self.fail('problem link checking sphinx docs:\\n'+output)


if __name__ == '__main__':
    unittest.main()
