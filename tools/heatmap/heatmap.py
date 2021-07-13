#!/usr/bin/env python

import sys
import argparse
from osgeo import osr
from osgeo import gdal

sys.path.append('../../model')
import cvig_fov as cvig
Globals = cvig.Globals

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


class ResizeSurface(object):
    """
    Resize surface photo to fit model and crop to fov.
    """
    def __init__(self, fov=360, panorama=True, random_orientation=True):
        self.transform = cvig.Resize(fov, panorama, random_orientation)
    def __call__(self, data):
        data_renamed = {'surface':data['image'], 'overhead':None}
        data = self.transform(data_renamed)
        return data


class ResizeOverhead(object):
    """
    Resize overhead image tile to fit model and crop to fov.
    """
    def __call__(self, data):
        data['image'] = torchvision.transforms.functional.resize(data['overhead'], (Globals.overhead_size, Globals.overhead_size))
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


class PolarTransform(object):
    def __init__(self):
        self.transform = cvig.PolarTransform()
    def __call__(self, data):
        data_renamed = {'overhead':data['image']}
        data = self.transform(data_renamed)
        return data


def sweep(aoi, bounds, edge, offset, sat_dir, photo_path, csv_path, temp_dir):

    # Load satellite strip
    sat_path = os.path.join(sat_dir, names[aoi-1] + '.tif')
    sat_file = gdal.Open(sat_path)

    # Specify transformations
    surface_transform = torchvision.transforms.Compose([
        Resize(fov, False),
        ImageNormalization(),
        PolarTransform()
    ])

    # Load models


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
                        default=(447665.8, 5411563.0, 448184.8, 5411814.8),
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
    parser.add_argument('-t', '--tempdir',
                        default='/local_data/geoloc/sat/temp',
                        help='Folder to hold temporary files')
    args = parser.parse_args()
    sweep(args.a, args.b, args.e, args.o, args.s, args.p, args.c, args.t)
