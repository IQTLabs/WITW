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
    Convert an image file to a batch of size 1
    """
    raw = io.imread(path)
    image = torch.from_numpy(raw.transpose((2, 0, 1)))
    batch = image.unsqueeze(0)
    return batch


def batch_to_file(batch, path):
    """
    Convert a batch of size 1 to an image file
    """
    image = torch.squeeze(batch, dim=0)
    raw = image.numpy().transpose((1, 2, 0))
    io.imsave(path, raw, check_contrast=False)


def main(options, surface_in, overhead_in, surface_out, overhead_out):

    # Setup
    surface_names = os.listdir(surface_in)
    overhead_names = os.listdir(overhead_in)
    names = sorted(list(set(surface_names).intersection(overhead_names)))
    os.makedirs(surface_out, exist_ok=True)
    os.makedirs(overhead_out, exist_ok=True)
    torch.no_grad = True

    # Define various constants
    aspect_model = np.array([[.02, 1., 9./16.],
                             [.12, 1., 2./3.],
                             [.13, 1., 3./4.],
                             [.05, 1., 1.],
                             [.30, 3./4., 1.],
                             [.33, 2./3., 1.],
                             [.05, 9./16., 1.]])
    aspect_cumsum = np.cumsum(aspect_model[:, 0])

    # Loop through image pairs
    for name in tqdm.tqdm(names):
        surface = file_to_batch(os.path.join(surface_in, name))
        overhead = file_to_batch(os.path.join(overhead_in, name))

        _, _, surface_height, surface_width = surface.size()
        _, _, overhead_height, overhead_width = overhead.size()
        surface_extend = surface.repeat([1,1,1,2])

        # Execute specified options
        if 10 in options:
            # Given a panorama, return a randomly-oriented slice of
            # hard-wired horizontal angular size.
            fov = 70
            width = round(fov / 360 * surface_width)
            start = torch.randint(surface_width, ())
            surface = torchvision.transforms.functional.crop(
                surface_extend, 0, start, surface_height, width)
        if 20 in options:
            # Given a panorama, return a mix of zoom/orientation/aspect ratio
            # combinations similar to a collection of real photographs.
            fov_min = 30.
            fov_max = 60.
            aov_degrees = fov_min + (fov_max - fov_min) * torch.rand(()).item()
            aov_pixels = aov_degrees / 360 * surface_width
            aspect_index = np.argmax(aspect_cumsum > torch.rand(()).item())
            height = round(aov_pixels * aspect_model[aspect_index, 1])
            width = round(aov_pixels * aspect_model[aspect_index, 2])
            left = torch.randint(surface_width, ()).item()
            vert_center = (surface_height - height) / 2
            vert_range = min(height / 3, surface_height - height)
            top = round(vert_center + (torch.rand(()).item() - 0.5) * vert_range)
            surface = torchvision.transforms.functional.crop(
                surface_extend, top, left, height, width)

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
