# -*- coding: utf-8 -*-

import os
import sys
import tempfile
import unittest

import numpy as np

import tables
from tables import utilsExtension


class Description1(tables.IsDescription):

    x = tables.Float64Col(dflt=1, shape=(2, 2))
    y = tables.UInt8Col(dflt=1)

    class info(tables.IsDescription):
        w = tables.StringCol(itemsize=2)
        x = tables.ComplexCol(itemsize=16)
        y = tables.Float64Col(dflt=1)
        z = tables.UInt8Col(dflt=1)

        class info2(tables.IsDescription):
            a = tables.EnumCol({'r':4, 'g':2, 'b':1}, 'r', 'int32', shape=2)


class DummyClass(object):
    pass

class CreateNestedTypeTestCase(unittest.TestCase):

    def setUp(self):
        self.opened_types = []
        self.fid = tempfile.mktemp()
        self.fileh = tables.openFile(self.fid, 'w')

    def tearDown(self):
        for type_id in self.opened_types:
            utilsExtension.CloseHDF5Type(type_id)
        self.fileh.close()
        os.remove(self.fid)

    def test_invalid_v_itemsize(self):
        desc = DummyClass()
        desc._v_itemsize = 0
        self.assertRaises(TypeError, utilsExtension.createNestedType, desc,
                          'little')

    # verify that createNestedType and createNestedTypeMatchByteorder
    # produce equivalent HDF5 types
    def test_individual_dtypes(self):
        dt_codes = ['i2', 'u2', 'i4', 'u4', 'i8', 'u8', 'f2', 'f4', 'f8',
                    'c8', 'c16', 'b1', 'i1', 'u1', 'S3']
        for dt_code in dt_codes:
            dtype = np.format_parser([dt_code], [], []).dtype
            table = self.fileh.createTable('/', dt_code, dtype)
            description = table.description
            tid1 = utilsExtension.createNestedType(description, sys.byteorder)
            tid2 = utilsExtension.createNestedTypeMatchByteorder(description,
                                                                 table.dtype)
            self.opened_types.extend([tid1, tid2])
            self.assertTrue(utilsExtension.HDF5TypesEqual(tid1, tid2))

    # convert to an HDF5 datatype, then back to a NumPy datatype
    def test_individual_dtypes_round_trip(self):
        # this doesn't work for 'b1', not sure if that's a bug...
        dt_codes = ['i2', 'u2', 'i4', 'u4', 'i8', 'u8', 'f2', 'f4', 'f8',
                    'c8', 'c16', 'i1', 'u1', 'S3']
        for dt_code in dt_codes:
            dtype = np.format_parser([dt_code], [], []).dtype
            table = self.fileh.createTable('/', dt_code, dtype)
            description = table.description
            tid1 = utilsExtension.createNestedType(description, sys.byteorder)
            self.opened_types.append(tid1)
            dtype2, _ = utilsExtension.HDF5ToNPExtType(tid1)
            self.assertTrue(isinstance(dtype2, np.dtype))
            self.assertEqual(dtype, dtype2)

    def test_multiple_non_nested_columns(self):
        dtype = np.format_parser(['i2', 'u2', 'i4', 'u4', 'i8', 'u8', 'f2',
                                  'f4', 'f8', 'c8', 'c16', 'b1', 'i1', 'u1',
                                  'S3'], [], []).dtype
        table = self.fileh.createTable('/', 'table', dtype)
        description = table.description
        tid1 = utilsExtension.createNestedType(description, sys.byteorder)
        tid2 = utilsExtension.createNestedTypeMatchByteorder(description,
                                                             table.dtype)
        self.opened_types.extend([tid1, tid2])
        self.assertTrue(utilsExtension.HDF5TypesEqual(tid1, tid2))

    def test_nested_columns(self):
        table = self.fileh.createTable('/', 'table', Description1)
        description = table.description
        tid1 = utilsExtension.createNestedType(description, sys.byteorder)
        tid2 = utilsExtension.createNestedTypeMatchByteorder(description,
                                                             table.dtype)
        self.opened_types.extend([tid1, tid2])
        self.assertTrue(utilsExtension.HDF5TypesEqual(tid1, tid2))


def suite():
    theSuite = unittest.TestSuite()
    theSuite.addTest(unittest.makeSuite(CreateNestedTypeTestCase))
    return theSuite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
