#!/usr/bin/env python

"""
Fraction of pixels with zeros in all bands
"""

import sys
import numpy as np
import gdalnumeric

if len(sys.argv) <= 1:
    raise Exception('! File path argument required')
data = gdalnumeric.LoadFile(sys.argv[1])
zero = np.prod(data == 0, axis=0)
frac = np.sum(zero) / zero.size
print(frac)
