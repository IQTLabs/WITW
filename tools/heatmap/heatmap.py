#!/usr/bin/env python

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--aoi',
                        type=int,
                        choices=range(1,12),
                        default=3,
                        help='SpaceNet AOI of satellite image')
    parser.add_argument('-b', '--bounds',
                        type=float,
                        nargs=4,
                        default=(447665.8, 5411563.0, 448184.8, 5411814.8)
                        metavar=('left', 'bottom', 'right', 'top'),
                        help='Bounds given as UTM coordinates in this order: min easting, min northing, max easting, max northing')
    args = parser.parse_args()
