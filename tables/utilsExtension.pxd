# -*- coding: utf-8 -*-

########################################################################
#
#       License: BSD
#       Created: March 03, 2008
#       Author:  Francesc Alted - faltet@pytables.com
#
#       $Id: definitions.pyd 1018 2005-06-20 09:43:34Z faltet $
#
########################################################################

"""
These are declarations for functions in utilsExtension.pyx that have to
be shared with other extensions.
"""

from definitions cimport hsize_t, hid_t

cdef hsize_t *malloc_dims(object)
cdef hid_t get_native_type(hid_t) nogil

cpdef hid_t AtomToHDF5Type(atom, str byteorder) except -1
cpdef hid_t createNestedType(object desc, str byteorder) except -1
cpdef hid_t createNestedTypeMatchByteorder(object desc, object dtype) except -1
