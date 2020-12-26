#!/usr/bin/env python

"""
Cross-View Image Geolocalization (CVIG)
2020-12-21 ~ D.P.Hogan
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from skimage import io

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class ImagePairDataset(torch.utils.data.Dataset):
    """
    Load pairs of images (one surface and one overhead)
    from paths specified in a CSV file.
    """
    def __init__(self, csv_path, base_path=None, transform=None):
        """
        Arguments:
        csv_path: Path to CSV file containing image paths.  File format:
            surface_file.tif,overhead_file.tif
        base_path: Starting folder for any relative file paths,
            if different from the folder containing the CSV file.
        """
        self.csv_path = csv_path
        if base_path is not None:
            self.base_path = base_path
        else:
            self.base_path = os.path.dirname(csv_path)
        self.transform = transform

        # Read file paths and convert any relative file paths to absolute
        file_paths = pd.read_csv(self.csv_path, names=['surface', 'overhead'])
        self.file_paths = file_paths.applymap(lambda x: os.path.join(self.base_path, x) if isinstance(x, str) and len(x)>0 and x[0] != '/' else x)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        surface_path = self.file_paths.iloc[idx]['surface']
        overhead_path = self.file_paths.iloc[idx]['overhead']
        surface_raw = io.imread(surface_path)
        overhead_raw = io.imread(overhead_path)
        surface = torch.from_numpy(surface_raw.astype(np.float32).transpose((2, 0, 1)))
        overhead = torch.from_numpy(overhead_raw.astype(np.float32).transpose((2, 0, 1)))
        if self.transform is not None:
            pass #Apply data augmentation here

        # Fix for CVUSA: stretch surface image to fit architecture
        surface = torch.repeat_interleave(surface, 2, dim=1)

        return {'surface':surface, 'overhead':overhead}


class SurfaceEncoder(nn.Module):
    def __init__(self, p=3.):
        super().__init__()
        self.bands = 3
        self.p = p
        self.conv_kwargs = {'kernel_size':4, 'stride':2, 'padding':0}
        self.activation = nn.LeakyReLU(0.2)
        self.bn_kwargs = {'momentum':0.1, 'affine':True, 'track_running_stats':True}
        
        self.conv1 = nn.Conv2d(self.bands, 64, **self.conv_kwargs)
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


OverheadEncoder = SurfaceEncoder


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


def train(csv_path = '/local_data/cvusa/train.csv', val_quantity=1000, batch_size=12, num_epochs=999999):

    # Data augmentation
    transform = None
    
    # Source the training and validation data
    trainval_set = ImagePairDataset(csv_path=csv_path, transform=transform)
    train_set, val_set = torch.utils.data.random_split(trainval_set, [len(trainval_set) -  val_quantity, val_quantity])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)

    # Neural networks
    surface_encoder = SurfaceEncoder().to(device)
    overhead_encoder = OverheadEncoder().to(device)
    if torch.cuda.device_count() > 1:
        surface_encoder = nn.DataParallel(surface_encoder)
        overhead_encoder = nn.DataParallel(overhead_encoder)
    # Loss function
    loss_func = exhaustive_minibatch_triplet_loss
    # Optimizer
    all_params = list(surface_encoder.parameters()) \
                 + list(overhead_encoder.parameters())
    optimizer = torch.optim.Adam(all_params)

    # Loop through epochs
    best_loss = None
    for epoch in range(num_epochs):
        print('Epoch %d' % (epoch + 1))

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

            print('  %5s: avg loss = %f' % (phase, running_loss / running_count))

        # Save weights if this is the lowest observed validation loss
        if best_loss is None or running_loss / running_count < best_loss:
            print('-------> new best')
            best_loss = running_loss / running_count
            torch.save(surface_encoder.state_dict(), './surface_best.pth')
            torch.save(overhead_encoder.state_dict(), './overhead_best.pth')


def test(csv_path = '/local_data/cvusa/test.csv', batch_size=12):

    # Source the test data
    test_set = ImagePairDataset(csv_path=csv_path, transform=None)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    # Load the neural network
    surface_encoder = SurfaceEncoder().to(device)
    overhead_encoder = OverheadEncoder().to(device)
    if torch.cuda.device_count() > 1:
        surface_encoder = nn.DataParallel(surface_encoder)
        overhead_encoder = nn.DataParallel(overhead_encoder)
    surface_encoder.load_state_dict(torch.load('./surface_best.pth'))
    overhead_encoder.load_state_dict(torch.load('./overhead_best.pth'))
    surface_encoder.eval()
    overhead_encoder.eval()

    # Loop through batches of data
    surface_embed = None
    overhead_embed = None
    for batch, data in enumerate(test_loader):
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
    top_one = 0
    top_five = 0
    top_ten = 0
    top_percent = 0
    rank_sum = 0
    count = surface_embed.size(0)
    for idx in range(count):
        this_surface_embed = torch.unsqueeze(surface_embed[idx, :], 0)
        distances = torch.pow(torch.sum(torch.pow(overhead_embed - this_surface_embed, 2), dim=1), 0.5)
        distance = distances[idx]
        rank = torch.sum(torch.le(distances, distance)).item()
        if rank <= 1:
            top_one += 1
        if rank <= 5:
            top_five += 1
        if rank <= 10:
            top_ten += 1
        if rank * 100 <= count:
            top_percent += 1
        rank_sum += rank

    # Print performance
    print('Top  1: {:.2f}%'.format(top_one / count * 100))
    print('Top  5: {:.2f}%'.format(top_five / count * 100))
    print('Top 10: {:.2f}%'.format(top_ten / count * 100))
    print('Top 1%: {:.2f}%'.format(top_percent / count * 100))
    print('Avg. Rank: {:.2f}'.format(rank_sum / count))
    print('Locations: {}'.format(count))


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1]=='train':
        train()
    elif sys.argv[1]=='test':
        test()
