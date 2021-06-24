#!/usr/bin/env python

"""
Count of nonzero values in all bands of an image
"""

import sys
import numpy as np
import gdalnumeric

if len(sys.argv) <= 1:
    print('! File path argument required')
data = gdalnumeric.LoadFile(sys.argv[1])
count = np.sum(data > 0)
print(count)
