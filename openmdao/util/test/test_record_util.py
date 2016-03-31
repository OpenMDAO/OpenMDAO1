import unittest
from openmdao.util.record_util import create_local_meta, update_local_meta, \
                                      format_iteration_coordinate


class RecordUtilTest(unittest.TestCase):
    def setUp(self):
        self.meta = create_local_meta(None, '')

    def test_create_meta(self):
        self.assertEqual(self.meta['name'], '')
        self.assertEqual(self.meta['coord'], [0, '', (0,)])

    def test_create_sub_meta(self):
        name = 'Testing'

        meta = create_local_meta(self.meta, name)

        self.assertEqual(meta['name'], name)
        self.assertEqual(meta['coord'], [0, '', (0,), name, (0,)])

    def test_update_meta(self):
        coord = (1, 2, 3)

        update_local_meta(self.meta, coord)
        self.assertEqual(self.meta['name'], '')
        self.assertEqual(self.meta['coord'], [0, '', coord])

    def test_format_coord(self):
        name = 'Sub'
        coord = (1, 2, 3)

        meta = create_local_meta(self.meta, name)
        update_local_meta(meta, coord)

        s = format_iteration_coordinate(meta['coord'])

        self.assertEqual(s, 'rank0:/0/Sub/1-2-3')

    def test_root_name(self):
        meta1 = create_local_meta(None, 'Driver')
        meta2 = create_local_meta(meta1, '')

        self.assertEqual(meta2['coord'], [0, 'Driver', (0,), 'root', (0,)])

    def test_integer_coord(self):

        update_local_meta(self.meta, 2)

        self.assertEqual(self.meta['coord'], [0, '', (2,)])
