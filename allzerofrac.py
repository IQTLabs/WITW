#!/usr/bin/env python

"""
Fraction of pixels with zeros in all bands
"""

import sys
import numpy as np
import gdalnumeric

if len(sys.argv) <= 1:
    raise Exception('! File path argument(s) required')
for path in sys.argv[1:]:
    data = gdalnumeric.LoadFile(path)
    zero = np.prod(data == 0, axis=0)
    frac = np.sum(zero) / zero.size
    print(path, frac)
