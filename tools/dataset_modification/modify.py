#!/usr/bin/env python

import os
#import sys
import argparse

#sys.path.append('../../model')
#import cvig_fov as cvig
#Globals = cvig.Globals
#device = cvig.device


def main(option, surface_in, overhead_in, surface_out, overhead_out):

    # Set up input
    surface_paths = os.listdir(surface_in)
    overhead_paths = os.listdir(overhead_in)

    # Set up output
    os.makedirs(surface_out, exist_ok=True)
    os.makedirs(overhead_out, exist_ok=True)

if __name__ == '__main__':

    default_surface_in = '/local_data/cvusa/bingmap/19'
    default_overhead_in = '/local_data/cvusa/streetview/panos'
    default_surface_out = '/local_data/geoloc/cvusamod/bingmap/19'
    default_overhead_out = '/local_data/geoloc/cvusamod/streetview/panos'

    parser = argparse.ArgumentParser()
    parser.add_argument('option',
                        default=0,
                        help='Image modification option')
    parser.add_argument('--surface_in',
                        default=default_surface_in,
                        help='Source directory for surface images')
    parser.add_argument('--overhead_in',
                        default=default_overhead_in,
                        help='Source directory for overhead images')
    parser.add_argument('--surface_out',
                        default=default_surface_out,
                        help='Output directory for surface images')
    parser.add_argument('--overhead_out',
                        default=default_overhead_out,
                        help='Output directory for overhead images')
    args = parser.parse_args()
    main(args.option, args.surface_in, args.overhead_in,
         args.surface_out, args.overhead_out)
