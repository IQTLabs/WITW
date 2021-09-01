#!/usr/bin/env python

import os
import argparse
from skimage import io

def file_to_batch(path):
    """
    Convert a single image file to batch of size 1
    """
    raw = io.imread(path)
    image = torch.from_numpy(raw.astype(np.float32).transpose((2, 0, 1)))
    batch = torch.unsqueeze(0) # Convert image to batch of size 1
    return batch

def batch_to_file(batch, path):
    """
    Convert a batch of size 1 to image file
    """
    pass


def main(options, surface_in, overhead_in, surface_out, overhead_out):

    # Setup
    surface_names = os.listdir(surface_in)
    overhead_names = os.listdir(overhead_in)
    os.makedirs(surface_out, exist_ok=True)
    os.makedirs(overhead_out, exist_ok=True)

    # Loop through views, then through images
    # Execute selected options
    for view in ['surface', 'overhead']:
        if view == 'surface':
            names = surface_names
            dir_in = surface_in
            dir_out = surface_out
        else:
            names = overhead_names
            dir_in = overhead_in
            dir_out = overhead_out
        for name in names:
            path_in = os.path.join(dir_in, name)
            path_out = os.path.join(dir_out, name)
            print(path_in)
            print(path_out)
            print()



    """
    for option in options:
        print('Option', option)
        if option == 0: # Filler; does nothing
            pass
        elif option == 1:
            pass
    """


if __name__ == '__main__':

    default_surface_in = '/local_data/cvusa/bingmap/19'
    default_overhead_in = '/local_data/cvusa/streetview/panos'
    default_surface_out = '/local_data/geoloc/cvusamod/bingmap/19'
    default_overhead_out = '/local_data/geoloc/cvusamod/streetview/panos'

    parser = argparse.ArgumentParser()
    parser.add_argument('options',
                        nargs='*',
                        type=int,
                        help='Image modification options')
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
    main(args.options, args.surface_in, args.overhead_in,
         args.surface_out, args.overhead_out)
