#!/usr/bin/env python

import os
import sys
import glob
import math
import time
import tqdm
import pathlib
import argparse
import numpy as np
import pandas as pd
from skimage import io

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device_parallel = False
device_ids = None

class Globals:
    dataset_paths = {
        'cvusa': {
            'train': './data/train-19zl.csv',
            'test': './data/val-19zl.csv'
        },
        'witw': {
            'train':'./data2/train.csv',
            'test':'./data2/test.csv'
        }
    }
    path_formats = {
        'cvusa': {
            'path_columns' : [0, 1],
            'path_names' : ['overhead', 'surface'],
            'header' : None,
            'panorama' : True,
        },
        'witw': {
            'path_columns' : [15, 16],
            'path_names' : ['surface', 'overhead'],
            'header' : 0,
            'panorama' : False,
        }
    }


class ImagePairDataset(torch.utils.data.Dataset):
    """
    Load pairs of images (one surface and one overhead)
    from paths specified in a CSV file.
    """
    def __init__(self, dataset, csv_path, base_path=None, transform=None):
        """
        Arguments:
        dataset: String specifying dataset ('cvusa' or 'witw')
        csv_path: Path to CSV file containing image paths.  File format:
            surface_file.tif,overhead_file.tif
        base_path: Starting folder for any relative file paths,
            if different from the folder containing the CSV file
        transform: transformation to apply, if any
        """
        self.csv_path = csv_path
        if base_path is not None:
            self.base_path = base_path
        else:
            self.base_path = os.path.dirname(csv_path)
        self.transform = transform

        # Read file paths and convert any relative file paths to absolute
        path_format = Globals.path_formats[dataset]
        file_paths = pd.read_csv(self.csv_path, header=path_format['header'], names=path_format['path_names'], usecols=path_format['path_columns'])
        self.file_paths = file_paths.applymap(lambda x: os.path.join(self.base_path, x) if isinstance(x, str) and len(x)>0 and x[0] != '/' else x)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load data
        surface_path = self.file_paths.iloc[idx]['surface']
        overhead_path = self.file_paths.iloc[idx]['overhead']
        surface_raw = io.imread(surface_path)
        overhead_raw = io.imread(overhead_path)
        surface = torch.from_numpy(surface_raw.astype(np.float32).transpose((2, 0, 1)))
        overhead = torch.from_numpy(overhead_raw.astype(np.float32).transpose((2, 0, 1)))
        data = {'surface':surface, 'overhead':overhead}

        # Transform data
        if self.transform is not None:
            data = self.transform(data)
        return data


def horizontal_shift(img, shift, unit='pixels'):
    """
    Shift a 360-degree surface panorama counterclockwise (i.e., as if the
    viewer were turning in a clockwise direction) by the specified amount.
    """
    if unit.lower() in ['pixels', 'pixel', 'p']:
        pix_shift = -round(shift)
    elif unit.lower() in ['fraction', 'fractions', 'f']:
        pix_shift = -round(shift * img.size(-1))
    elif unit.lower() in ['degrees', 'degree', 'd']:
        pix_shift = -round(shift * img.size(-1) / 360.)
    elif unit.lower() in ['radians', 'radian', 'r']:
        pix_shift = -round(shift * img.size(-1) / (2 * math.pi))
    else:
        raise Exception('! Invalid unit in horizontal_shift()')
    return torch.roll(img, pix_shift, dims=-1)


def quantized_rotation(img, factor):
    """
    Rotate an image counterclockwise by an integer factor times 90 degrees.
    """
    if factor % 4 == 0:
        pass
    elif factor % 4 == 1:
        img = img.transpose(-2, -1).flip(-1)
    elif factor % 4 == 2:
        img = img.flip(-2).flip(-1)
    elif factor % 4 == 3:
        img = img.transpose(-2, -1).flip(-2)
    return img


class SyncedRotation(object):
    """
    Rotate the overhead image by a random angle.  If the surface image
    is a panorama, shift it by the corresponding amount.
    """
    def __init__(self, dataset):
        self.dataset = dataset
    def __call__(self, data):
        angle = torch.rand(()).item() * 360.
        if Globals.path_formats[self.dataset]['panorama']:
            data['surface'] = horizontal_shift(
                data['surface'], angle, unit='degrees')
        data['overhead'] = torchvision.transforms.functional.rotate(
            data['overhead'], angle)
        return data


class QuantizedSyncedRotation(object):
    """
    Rotate the overhead image by a random multiple of 90 degrees.  If the
    surface image is a panorama, shift it by the corresponding amount.
    """
    def __init__(self, dataset):
        self.dataset = dataset
    def __call__(self, data):
        factor = torch.randint(4, ()).item()
        if Globals.path_formats[self.dataset]['panorama']:
            data['surface'] = horizontal_shift(
                data['surface'], factor * 90, unit='degrees')
        data['overhead'] = quantized_rotation(data['overhead'], factor)
        return data


# def orientation_map(x, view='surface', orientation_dict=None):
#     """
#     Returns "orientation map" tensor.  See Liu & Li CVPR 2019.

#     Arguments:
#     x = input image tensor
#     view = 'surface' or 'overhead'
#     orientation_dict = a dictionary in which to save previous results
#         for possible reuse (this improves performance)
#     """
#     if orientation_dict is not None:
#         description = (view, x.shape, x.device)
#     if orientation_dict is not None and description in orientation_dict.keys():
#         return orientation_dict[description]
#     else:
#         shape = (x.size(-2), x.size(-1))
#         shape_expanded = np.expand_dims(np.array(shape), (1,2))
#         shape_max = max(shape)
#         uv = np.indices(shape, dtype=float)
#         uv = (2 * uv - shape_expanded + 1) / (shape_max - 1)
#         if view == 'overhead':
#             uv[0], uv[1] = (np.sqrt(uv[0]**2 + uv[1]**2) / math.sqrt(2)) \
#                            * 2. - 1., \
#                            np.arctan2(uv[1], -uv[0]) / math.pi
#         uv = torch.tensor(uv, dtype=torch.float).to(x.device)
#         if orientation_dict is not None:
#             orientation_dict[description] = uv
#         return uv


# class OrientationMaps(object):
#     """
#     Applies appropriate "orientation map" tensors to surface/overhead pair.
#     See Liu & Li CVPR 2019.
#     """
#     orientation_dicts = [{}, {}]

#     def __call__(self, data):
#         for view, od in zip(['surface', 'overhead'], self.orientation_dicts):
#             x = data[view]
#             uv = orientation_map(data[view], view, od)
#             cat = torch.cat((x, uv), dim=0)
#             data[view] = cat
#         return data


class SurfaceResize(object):
    """
    Resize surface image to fit this model architecture.
    """
    def __init__(self, dataset):
        self.dataset = dataset
    def __call__(self, data):
        if self.dataset == 'cvusa':
            data['surface'] = torch.repeat_interleave(
                data['surface'], 2, dim=-2)
        elif self.dataset == 'witw':
            data['surface'] = torchvision.transforms.functional.resize(
                data['surface'], [500, 500])
        else:
            raise Exception('! Invalid dataset type in ' + type(self).__name__
                            + '().')
        return data


class SurfaceEncoder(nn.Module):
    def __init__(self, orientation=False, bands=3, p=3.):
        super().__init__()
        self.orientation = orientation
        self.bands = bands
        self.p = p
        self.inputs = self.bands + 2 * self.orientation
        self.conv_kwargs = {'kernel_size':4, 'stride':2, 'padding':0}
        self.activation = nn.LeakyReLU(0.2)
        self.bn_kwargs = {'momentum':0.1, 'affine':True,
                          'track_running_stats':True}

        self.conv1 = nn.Conv2d(self.inputs, 64, **self.conv_kwargs)
        self.bn1 = nn.BatchNorm2d(64, **self.bn_kwargs)
        self.conv2 = nn.Conv2d(64, 128, **self.conv_kwargs)
        self.bn2 = nn.BatchNorm2d(128, **self.bn_kwargs)
        self.conv3 = nn.Conv2d(128, 256, **self.conv_kwargs)
        self.bn3 = nn.BatchNorm2d(256, **self.bn_kwargs)
        self.conv4 = nn.Conv2d(256, 512, **self.conv_kwargs)
        self.bn4 = nn.BatchNorm2d(512, **self.bn_kwargs)
        self.conv5 = nn.Conv2d(512, 512, **self.conv_kwargs)
        self.bn5 = nn.BatchNorm2d(512, **self.bn_kwargs)
        self.conv6 = nn.Conv2d(512, 512, **self.conv_kwargs)
        self.bn6 = nn.BatchNorm2d(512, **self.bn_kwargs)
        self.conv7 = nn.Conv2d(512, 512, **self.conv_kwargs)
        self.bn7 = nn.BatchNorm2d(512, **self.bn_kwargs)

        def set_initial_weights(module):
            if type(module) == nn.Conv2d:
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                torch.nn.init.normal_(module.bias, mean=0.0, std=0.02)
            elif type(module) == nn.BatchNorm2d:
                torch.nn.init.normal_(module.weight, mean=1.0, std=0.02)
                torch.nn.init.normal_(module.bias, mean=0.0, std=0.02)
        self.apply(set_initial_weights)

    def forward(self, x):
        x = x / 255.
        x = -1. + 2. * x
        x = self.bn1(self.activation(self.conv1(x)))
        x = self.bn2(self.activation(self.conv2(x)))
        x = self.bn3(self.activation(self.conv3(x)))
        x = self.bn4(self.activation(self.conv4(x)))
        x = self.bn5(self.activation(self.conv5(x)))
        f1 = torch.pow(torch.mean(torch.pow(F.relu(x), self.p), [2, 3]), 1./self.p)
        x = self.bn6(self.activation(self.conv6(x)))
        f2 = torch.pow(torch.mean(torch.pow(F.relu(x), self.p), [2, 3]), 1./self.p)
        x = self.bn7(self.activation(self.conv7(x)))
        f3 = torch.pow(torch.mean(torch.pow(F.relu(x), self.p), [2, 3]), 1./self.p)
        f = torch.cat((f1, f2, f3), 1)
        f = f / torch.unsqueeze(torch.pow(torch.linalg.norm(f, dim=1), 0.5), 1)
        return f


class OverheadEncoder(SurfaceEncoder):
    pass


def exhaustive_minibatch_triplet_loss(embed1, embed2, soft_margin=False, alpha=10., margin=1.):
    """
    Arguments:
        embed1: Minibatch of embeddings.  Shape = (batch size, dimensionality
            of feature vector)
        embed2: Corresponding embeddings, such as from the other branch of a
            Siamese network
        soft_margin: False for hard-margin triplet loss; true for soft-margin
            triplet loss
        alpha: Parameter used by soft-margin triplet loss
        margin: Parameter used by hard-margin triplet loss
    Output:
        Triplet loss, using all valid combinations (exhaustive minibatch
        strategy) and squared Euclidean distances
    See arXiv:1608.00161 Section 5.3 and Liu & Li CVPR 2019 Sections 4.3, 5
    """
    loss = torch.tensor(0.)
    batch_size = embed1.size(0)
    for (a, p) in [(embed1, embed2), (embed2, embed1)]:
        for shift in range(1, batch_size):
            n = torch.roll(p, shift, dims=0)
            ap_dist2 = torch.sum((p - a)**2, dim=1)
            an_dist2 = torch.sum((n - a)**2, dim=1)
            if soft_margin:
                this_loss = torch.log(1. + torch.exp(alpha * (ap_dist2 - an_dist2)))
            else:
                this_loss = F.relu(ap_dist2 - an_dist2 + margin)
            loss = loss + torch.sum(this_loss)
    loss = loss / (2*batch_size*(batch_size-1))
    return loss


def train(dataset='cvusa', val_quantity=1000, batch_size=16, num_workers=4, num_epochs=999999):

    pathlib.Path('./weights').mkdir(parents=True, exist_ok=True)
    csv_path = Globals.dataset_paths[dataset]['train']

    # Data modification and augmentation
    transform = torchvision.transforms.Compose([
        SyncedRotation(dataset),
        #OrientationMaps(),
        SurfaceResize(dataset)
    ])

    # Source the training and validation data
    trainval_set = ImagePairDataset(dataset=dataset, csv_path=csv_path, transform=transform)
    train_set, val_set = torch.utils.data.random_split(trainval_set, [len(trainval_set) - val_quantity, val_quantity])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)

    # Neural networks
    surface_encoder = SurfaceEncoder().to(device)
    overhead_encoder = OverheadEncoder().to(device)
    if device_parallel and torch.cuda.device_count() > 1:
        surface_encoder = nn.DataParallel(surface_encoder,
                                          device_ids=device_ids)
        overhead_encoder = nn.DataParallel(overhead_encoder,
                                           device_ids=device_ids)
    # Loss function
    loss_func = exhaustive_minibatch_triplet_loss
    # Optimizer
    all_params = list(surface_encoder.parameters()) \
                 + list(overhead_encoder.parameters())
    optimizer = torch.optim.Adam(all_params)

    # Loop through epochs
    best_loss = None
    for epoch in range(num_epochs):
        print('Epoch %d, %s' % (epoch + 1, time.ctime(time.time())))

        for phase in ['train', 'val']:
            running_count = 0
            running_loss = 0.

            if phase == 'train':
                loader = train_loader
                surface_encoder.train()
                overhead_encoder.train()
            elif phase == 'val':
                loader = val_loader
                surface_encoder.eval()
                overhead_encoder.eval()

            #Loop through batches of data
            for batch, data in enumerate(loader):
                surface = data['surface'].to(device)
                overhead = data['overhead'].to(device)

                with torch.set_grad_enabled(phase == 'train'):

                    # Forward and loss (train and val)
                    surface_embed = surface_encoder(surface)
                    overhead_embed = overhead_encoder(overhead)
                    loss = loss_func(surface_embed, overhead_embed)

                    # Backward and optimization (train only)
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                count = surface_embed.size(0)
                running_count += count
                running_loss += loss.item() * count

                print('epoch = {} {}, iter = {}, count = {}, loss = {:.4f}'.format(epoch+1, phase, batch, running_count, loss))

            print('  %5s: avg loss = %f' % (phase, running_loss / running_count))

        # Save weights if this is the lowest observed validation loss
        if best_loss is None or running_loss / running_count < best_loss:
            print('-------> new best')
            best_loss = running_loss / running_count
            torch.save(surface_encoder.state_dict(),
                       './weights/surface_best.pth')
            torch.save(overhead_encoder.state_dict(),
                       './weights/overhead_best.pth')


def test(dataset='cvusa', batch_size=16, num_workers=4):

    csv_path = Globals.dataset_paths[dataset]['test']

    # Specify transformation, if any
    transform = torchvision.transforms.Compose([
        SyncedRotation(dataset),
        #OrientationMaps(),
        SurfaceResize(dataset)
    ])

    # Source the test data
    test_set = ImagePairDataset(dataset=dataset, csv_path=csv_path, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)

    # Load the neural network
    surface_encoder = SurfaceEncoder().to(device)
    overhead_encoder = OverheadEncoder().to(device)
    if device_parallel and torch.cuda.device_count() > 1:
        surface_encoder = nn.DataParallel(
            surface_encoder, device_ids=device_ids)
        overhead_encoder = nn.DataParallel(
            overhead_encoder, device_ids=device_ids)
    surface_encoder.load_state_dict(torch.load(
        './weights/surface_best.pth', map_location='cpu'))
    overhead_encoder.load_state_dict(torch.load(
        './weights/overhead_best.pth', map_location='cpu'))
    surface_encoder.eval()
    overhead_encoder.eval()

    # Loop through batches of data
    surface_embed = None
    overhead_embed = None
    for batch, data in enumerate(tqdm.tqdm(test_loader)):
        surface = data['surface'].to(device)
        overhead = data['overhead'].to(device)

        with torch.set_grad_enabled(False):
            surface_embed_part = surface_encoder(surface)
            overhead_embed_part = overhead_encoder(overhead)

            if surface_embed is None:
                surface_embed = surface_embed_part
                overhead_embed = overhead_embed_part
            else:
                surface_embed = torch.cat((surface_embed, surface_embed_part), dim=0)
                overhead_embed = torch.cat((overhead_embed, overhead_embed_part), dim=0)

    # Measure performance
    count = surface_embed.size(0)
    ranks = np.zeros([count], dtype=int)
    for idx in tqdm.tqdm(range(count)):
        this_surface_embed = torch.unsqueeze(surface_embed[idx, :], 0)
        distances = torch.pow(torch.sum(torch.pow(overhead_embed - this_surface_embed, 2), dim=1), 0.5)
        distance = distances[idx]
        ranks[idx] = torch.sum(torch.le(distances, distance)).item()
    top_one = np.sum(ranks <= 1) / count * 100
    top_five = np.sum(ranks <= 5) / count * 100
    top_ten = np.sum(ranks <= 10) / count * 100
    top_percent = np.sum(ranks * 100 <= count) / count * 100
    mean = np.mean(ranks)
    median = np.median(ranks)

    # Print performance
    print('Top  1: {:.2f}%'.format(top_one))
    print('Top  5: {:.2f}%'.format(top_five))
    print('Top 10: {:.2f}%'.format(top_ten))
    print('Top 1%: {:.2f}%'.format(top_percent))
    print('Avg. Rank: {:.2f}'.format(mean))
    print('Med. Rank: {:.2f}'.format(median))
    print('Locations: {}'.format(count))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',
                        default='train',
                        choices=['train', 'test'],
                        help='Run mode. [Default = train]')
    parser.add_argument('--dataset',
                        default='cvusa',
                        choices=['cvusa', 'witw'],
                        help='Dataset to use. [Default = cvusa]')
    args = parser.parse_args()
    if args.mode == 'train':
        train(dataset=args.dataset)
    elif args.mode == 'test':
        test(dataset=args.dataset)
