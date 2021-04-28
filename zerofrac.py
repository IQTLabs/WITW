#!/usr/bin/env python

"""
Fraction of zeros in all bands of an image
"""

import sys
import numpy as np
import gdalnumeric

if len(sys.argv) <= 1:
    print('! File path argument required')
data = gdalnumeric.LoadFile(sys.argv[1])
frac = np.sum(data == 0) / data.size
print(frac)
