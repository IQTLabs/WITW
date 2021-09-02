#!/usr/bin/env python

import os
import argparse
import numpy as np
from skimage import io
import tqdm
import torch
import torchvision


def file_to_batch(path):
    """
    Convert a single image file to batch of size 1
    """
    raw = io.imread(path)
    #image = torch.from_numpy(raw.astype(np.float32).transpose((2, 0, 1)))
    image = torch.from_numpy(raw.transpose((2, 0, 1)))
    batch = image.unsqueeze(0)
    return batch


def batch_to_file(batch, path):
    """
    Convert a batch of size 1 to image file
    """
    image = torch.squeeze(batch, dim=0)
    raw = image.numpy().transpose((1, 2, 0))
    io.imsave(path, raw)
    #torchvision.utils.save_image(image, path, value_range=(0,255))


def main(options, surface_in, overhead_in, surface_out, overhead_out):

    # Setup
    surface_names = os.listdir(surface_in)
    overhead_names = os.listdir(overhead_in)
    names = sorted(list(set(surface_names).intersection(overhead_names)))
    os.makedirs(surface_out, exist_ok=True)
    os.makedirs(overhead_out, exist_ok=True)

    # Loop through image pairs
    for name in tqdm.tqdm(names):
        surface = file_to_batch(os.path.join(surface_in, name))
        overhead = file_to_batch(os.path.join(overhead_in, name))

        if 1 in options:
            batch_to_file(surface, os.path.join(surface_out, name))
        if 2 in options:
            batch_to_file(overhead, os.path.join(overhead_out, name))


if __name__ == '__main__':

    default_surface_in = '/local_data/cvusa/streetview/panos'
    default_overhead_in = '/local_data/cvusa/bingmap/19'
    default_surface_out = '/local_data/geoloc/cvusamod/streetview/panos'
    default_overhead_out = '/local_data/geoloc/cvusamod/bingmap/19'

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
