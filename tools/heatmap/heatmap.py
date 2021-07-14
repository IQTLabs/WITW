#!/usr/bin/env python

import os
import sys
import tqdm
import argparse
import numpy as np
from skimage import io
import torch
import torchvision
from osgeo import osr
from osgeo import gdal

sys.path.append('../../model')
import cvig_fov as cvig
Globals = cvig.Globals
device = cvig.device

names = [
    '01_rio',
    '02_vegas',
    '03_paris',
    '04_shanghai',
    '05_khartoum',
    '06_atlanta',
    '07_moscow',
    '08_mumbai',
    '09_san',
    '10_dar',
    '11_rotterdam',
]


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        raw = io.imread(self.paths[idx])
        image = torch.from_numpy(raw.astype(np.float32).transpose((2, 0, 1)))
        data = {'image':image}
        if self.transform is not None:
            data = self.transform(data)
        return data


class TileDataset(torch.utils.data.Dataset):
    def __init__(self, source, windows, transform=None):
        self.source = source
        self.windows = windows
        self.transform = transform
    def __len__(self):
        return len(self.windows)
    def __getitem__(self, idx):
        mem_path = '/vsimem/tile%s.jpg' % str(idx)
        ds = gdal.Translate(mem_path, self.source, projWin=self.windows[idx])
        raw = ds.ReadAsArray()
        gdal.GetDriverByName('GTiff').Delete(mem_path)
        image = torch.from_numpy(raw.astype(np.float32))
        data = {'image':image}
        if self.transform is not None:
            data = self.transform(data)
        return data


class ResizeSurface(object):
    """
    Resize surface photo to fit model and crop to fov.
    """
    def __init__(self, fov=360):
        self.fov = fov
        self.surface_width = int(self.fov / 360 * Globals.surface_width_max)
    def __call__(self, data):
        data['image'] = torchvision.transforms.functional.resize(data['image'], (Globals.surface_height_max, self.surface_width))
        return data


class ResizeOverhead(object):
    """
    Resize overhead image tile to fit model and crop to fov.
    """
    def __call__(self, data):
        data['image'] = torchvision.transforms.functional.resize(data['image'], (Globals.overhead_size, Globals.overhead_size))
        return data


class ImageNormalization(object):
    """
    Normalize image values to use with pretrained VGG model
    """
    def __init__(self):
        self.norm = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    def __call__(self, data):
        data['image'] = self.norm(data['image'] / 255.)
        return data


class PolarTransform(object):
    def __init__(self):
        self.transform = cvig.PolarTransform()
    def __call__(self, data):
        data_renamed = {'overhead':data['image']}
        data = self.transform(data_renamed)
        return data


def sweep(aoi, bounds, edge, offset, fov, sat_dir, photo_path, csv_path):

    # Compute center and window for each satellite tile
    center_eastings = []
    center_northings = []
    windows = []
    for easting in np.arange(bounds[0], bounds[2], offset):
        for northing in np.arange(bounds[3], bounds[1], -offset):
            center_eastings.append(easting + edge / 2.)
            center_northings.append(northing - edge / 2.)
            windows.append([easting, northing, easting + edge, northing - edge])

    # Load satellite strip
    sat_path = os.path.join(sat_dir, names[aoi-1] + '.tif')
    sat_file = gdal.Open(sat_path)

    # Specify transformations
    surface_transform = torchvision.transforms.Compose([
        ResizeSurface(fov),
        ImageNormalization()
    ])
    overhead_transform = torchvision.transforms.Compose([
        ResizeOverhead(),
        ImageNormalization(),
        PolarTransform()
    ])

    # Load data
    surface_set = ImageDataset((photo_path,), surface_transform)
    overhead_set = TileDataset(sat_file, windows, overhead_transform)
    surface_batch = torch.unsqueeze(surface_set[0]['image'], dim=0).to(device)
    overhead_loader = torch.utils.data.DataLoader(overhead_set, batch_size=64, shuffle=False, num_workers=1)

    # Load the neural networks
    surface_encoder = cvig.FOV_DSM(circ_padding=False).to(device)
    overhead_encoder = cvig.FOV_DSM(circ_padding=True).to(device)
    surface_encoder.load_state_dict(torch.load('../../model/fov_{}_surface_best.pth'.format(int(fov))))
    overhead_encoder.load_state_dict(torch.load('../../model/fov_{}_overhead_best.pth'.format(int(fov))))
    surface_encoder.eval()
    overhead_encoder.eval()

    # Surface photo's features
    surface_embed = surface_encoder(surface_batch)

    # Overhead images' features
    overhead_embed = None
    for batch, data in enumerate(tqdm.tqdm(overhead_loader)):
        overhead = data['polar'].to(device)
        with torch.set_grad_enabled(False):
            overhead_embed_part = overhead_encoder(overhead)
            if overhead_embed is None:
                overhead_embed = overhead_embed_part
            else:
                overhead_embed = torch.cat((overhead_embed, overhead_embed_part), dim=0)

    # Calculate score for each overhead image
    output_width_max = 64
    orientation_estimate = cvig.correlation(overhead_embed, surface_embed)
    orientations = torch.squeeze(orientation_estimate) * 360 / output_width_max
    overhead_cropped_all = cvig.crop_overhead(overhead_embed, orientation_estimate, surface_embed.shape[3])
    distances = cvig.l2_distance(overhead_cropped_all, surface_embed)
    distances = torch.squeeze(distances)
    scores = torch.exp(10. * (1. - distances))
    print(orientations)
    print(distances)
    print(scores)

def layer(aoi, bounds, sat_dir, layer_path):
    sat_path = os.path.join(sat_dir, names[aoi-1] + '.tif')
    sat_file = gdal.Open(sat_path)
    window = [bounds[0], bounds[3], bounds[2], bounds[1]]
    gdal.Translate(layer_path, sat_file, projWin=window)


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
                        default=(447665.8, 5411486.8, 448184.8, 5411814.8),
                        metavar=('left', 'bottom', 'right', 'top'),
                        help='Bounds given as UTM coordinates in this order: min easting, min northing, max easting, max northing')
    parser.add_argument('-e', '--edge',
                        type=float,
                        default=225,
                        help='Edge length of satellite imagery tiles [m]')
    parser.add_argument('-o', '--offset',
                        type=float,
                        default=56.25,
                        help='Offset between centers of adjacent satellite imagery tiles [m]')
    parser.add_argument('-f', '--fov',
                        type=int,
                        default=70,
                        help='Field of view assumed for photo (deg, rounded)')
    parser.add_argument('-s', '--satdir',
                        default='/local_data/geoloc/sat/utm',
                        help='Folder containing satellite images')
    parser.add_argument('-p', '--photopath',
                        default='img.jpg',
                        help='Path to surface photo to analyze')
    parser.add_argument('-c', '--csvpath',
                        default='./geomatch.csv',
                        help='Output CSV file path')
    parser.add_argument('-l', '--layerpath',
                        default='./satlayer.tiff',
                        help='Output cropped satellite image')
    args = parser.parse_args()
    sweep(args.aoi, args.bounds, args.edge, args.offset, args.fov,
          args.satdir, args.photopath, args.csvpath)
    #layer(args.aoi, args.bounds, args.satdir, args.layerpath)
