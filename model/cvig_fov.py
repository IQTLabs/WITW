#!/usr/bin/env python

import argparse
import os
import sys
import math
import time
import tqdm
import numpy as np
import pandas as pd
from skimage import io

import torch
import torchvision

class Globals:
    surface_height_max = 128
    surface_width_max = 512
    overhead_size = 256

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
            'header' : None
        },
        'witw': {
            'path_columns' : [15, 16],
            'path_names' : ['surface', 'overhead'],
            'header' : 0
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


class ResizeCVUSA(object):
    """
    Resize the CVUSA images to fit model and crop to fov.
    """
    def __init__(self, fov=360, random_orientation=True):
        self.fov = fov
        self.surface_width = int(self.fov / 360 * Globals.surface_width_max)
        self.random_orientation = random_orientation

    def __call__(self, data):
        data['surface'] = torchvision.transforms.functional.resize(data['surface'], (Globals.surface_height_max, Globals.surface_width_max))
        if self.random_orientation:
            start = torch.randint(0, Globals.surface_width_max, ())
        else:
            start = 0
        end = start + self.surface_width
        if end < Globals.surface_width_max:
            data['surface'] = data['surface'][:,:,start:end]
        else:
            data['surface'] = torch.cat((data['surface'][:,:,start:], data[
                'surface'][:,:,:end - Globals.surface_width_max]), dim=2)
        data['overhead'] = torchvision.transforms.functional.resize(data['overhead'], (Globals.overhead_size, Globals.overhead_size))
        return data


class ImageNormalization(object):
    """
    Normalize image values to use with pretrained VGG model
    """
    def __init__(self):
        self.keys = ['surface', 'overhead']
        self.norm = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __call__(self, data):
        for key in self.keys:
            data[key] = self.norm(data[key] / 255.)
        return data


def bilinear_interpolate(im, x, y):
    # https://stackoverflow.com/a/12729229

    assert x.shape == y.shape
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[2]-1);
    x1 = np.clip(x1, 0, im.shape[2]-1);
    y0 = np.clip(y0, 0, im.shape[1]-1);
    y1 = np.clip(y1, 0, im.shape[1]-1);

    Ia = im[:, y0, x0 ]
    Ib = im[:, y1, x0 ]
    Ic = im[:, y0, x1 ]
    Id = im[:, y1, x1 ]

    wa = torch.FloatTensor(((x1-x) * (y1-y)).reshape(1, *x.shape))
    wb = torch.FloatTensor(((x1-x) * (y-y0)).reshape(1, *x.shape))
    wc = torch.FloatTensor(((x-x0) * (y1-y)).reshape(1, *x.shape))
    wd = torch.FloatTensor(((x-x0) * (y-y0)).reshape(1, *x.shape))

    return wa*Ia + wb*Ib + wc*Ic + wd*Id


class PolarTransform(object):
    """
    Applies polar transform from "Where am I looking at? Joint Location and Orientation Estimation by Cross-View Matching"
    CVPR 2020.
    """
    def __call__(self, data):
        h_s = Globals.surface_height_max
        w_s = Globals.surface_width_max
        s_o = Globals.overhead_size

        transf_overhead = torch.zeros((data['overhead'].size(0), h_s, w_s))
        xx, yy = np.meshgrid(range(w_s), range(h_s))
        yy_o = (s_o/2) + (s_o/2) * (h_s - 1 - yy)/h_s * np.cos(
            2 * math.pi * xx / w_s)
        xx_o = (s_o/2) - (s_o/2) * (h_s - 1 - yy)/h_s * np.sin(
            2 * math.pi * xx / w_s)
        #yy_o = np.floor(yy_o)
        #xx_o = np.floor(xx_o)
        #transf_overhead[:, yy.flatten(), xx.flatten()] = data['overhead'][
        #    :, yy_o.flatten(), xx_o.flatten()]
        transf_overhead = bilinear_interpolate(data['overhead'], xx_o, yy_o)

        data['polar'] = transf_overhead
        return data


class HorizCircPadding(torch.nn.Module):
    """
    Modify torch.nn.Conv2d layer to use circular padding in horizontal
    direction and zero padding in vertical direction, while retaining weights.
    """
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        padding = self.layer.padding
        # Vertical zero padding with prelayer
        self.prelayer = torch.nn.ConstantPad2d(
            (0, 0, padding[0], padding[0]), 0)
        # Horizontal circular padding with layer
        self.layer.padding = (0, padding[1])
        self.layer._reversed_padding_repeated_twice = torch.nn.modules.utils._reverse_repeat_tuple(self.layer.padding, 2)
        self.layer.padding_mode='circular'
    def forward(self, x):
        x = self.prelayer(x)
        x = self.layer(x)
        return x


class AddDropout(torch.nn.Module):
    """
    Modify torch.nn.Conv2d layer to add dropout, while retaining weights.
    """
    def __init__(self, layer, p=0.5):
        super().__init__()
        self.layer = layer
        self.postlayer = torch.nn.Dropout2d(p=p)
    def forward(self, x):
        x = self.layer(x)
        x = self.postlayer(x)
        return x


class FOV_DSM(torch.nn.Module):
    """
    Prepare vgg16 model with modification from "Where am I looking at? Joint Location and Orientation Estimation by Cross-View Matching"
    CVPR 2020.
    """
    def __init__(self, circ_padding=False):
        super(FOV_DSM, self).__init__()

        model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16', pretrained=True)
        # model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16_bn', pretrained=True)

        # shorten model based on Where Am I Looking modifications
        model.features = model.features[:23]

        model.features.add_module(str(len(model.features)), torch.nn.Conv2d(512, 256, 3, (2, 1), padding=1))
        torch.nn.init.xavier_uniform_(model.features[-1].weight)
        torch.nn.init.zeros_(model.features[-1].bias)
        model.features.add_module(str(len(model.features)), torch.nn.ReLU(inplace=True))
        model.features.add_module(str(len(model.features)), torch.nn.Conv2d(256, 64, 3, (2, 1), padding=1))
        torch.nn.init.xavier_uniform_(model.features[-1].weight)
        torch.nn.init.zeros_(model.features[-1].bias)
        model.features.add_module(str(len(model.features)), torch.nn.ReLU(inplace=True))
        model.features.add_module(str(len(model.features)), torch.nn.Conv2d(64, 16, 3, padding=1))
        torch.nn.init.xavier_uniform_(model.features[-1].weight)
        torch.nn.init.zeros_(model.features[-1].bias)

        # only train last 6 conv layers
        for name, param in model.features.named_parameters():
            torch_layer_num = int(name.split('.')[0])
            if torch_layer_num < 17:
                param.requires_grad = False

        # circular padding
        if circ_padding:
            for i, layer in enumerate(model.features):
                if isinstance(layer, torch.nn.Conv2d):
                    model.features[i] = HorizCircPadding(layer)

        # dropout
        for i in [17, 19, 21]:
            model.features[i] = AddDropout(model.features[i], 0.2)

        self.model = model

    def forward(self, x):
        out = self.model.features(x)
        return out


def correlation(overhead_embed, surface_embed):

    o_c, o_h, o_w = overhead_embed.shape[1:]
    s_c, s_h, s_w = surface_embed.shape[1:]
    assert o_h == s_h, o_c == s_c

    # append beginning of overhead embedding to the end to get a full correlation
    n = s_w - 1
    x = torch.cat((overhead_embed, overhead_embed[:, :, :, :n]), axis=3)
    f = surface_embed

    # calculate correlation using convolution
    out = torch.nn.functional.conv2d(x, f, stride=1)
    h, w = out.shape[-2:]
    assert h==1, w==o_w

    # get index of maximum correlation
    out = torch.squeeze(out, -2)
    orientation = torch.argmax(out,-1)  # shape = [batch_overhead, batch_surface]

    return orientation


def crop_overhead(overhead_embed, orientation, surface_width):
    batch_overhead, batch_surface = orientation.shape
    c, h, w = overhead_embed.shape[1:]
    # duplicate overhead embeddings according to batch size
    overhead_embed = torch.unsqueeze(overhead_embed, 1) # shape = [batch_overhead, 1, c, h, w]
    overhead_embed = torch.tile(overhead_embed, [1, batch_surface, 1, 1, 1]) # shape = [batch_overhead, batch_surface, c, h, w]
    orientation = torch.unsqueeze(orientation, -1) # shape = [batch_overhead, batch_surface, 1]

    # reindex overhead embeddings
    i = torch.arange(batch_overhead).to(device)
    j = torch.arange(batch_surface).to(device)
    k = torch.arange(w).to(device)
    x, y, z = torch.meshgrid(i, j, k)
    z_index = torch.fmod(z + orientation, w)
    overhead_embed = overhead_embed.permute(0,1,4,2,3)
    overhead_reindex = overhead_embed[x,y,z_index,:,:]
    overhead_reindex = overhead_reindex.permute(0,1,3,4,2)

    # crop overhead embeddings
    overhead_cropped = overhead_reindex[:,:,:,:,:surface_width] # shape = [batch_overhead, batch_surface, c, h, surface_width]
    assert overhead_cropped.shape[4] == surface_width

    return overhead_cropped


def l2_distance(overhead_cropped, surface_embed):

    # l2 normalize overhead embedding
    batch_overhead, batch_surface, c, h, overhead_width = overhead_cropped.shape
    overhead_normalized = overhead_cropped.reshape(batch_overhead, batch_surface, -1)
    overhead_normalized = torch.div(overhead_normalized, torch.linalg.norm(overhead_normalized, ord=2, dim=-1).unsqueeze(-1))
    overhead_normalized = overhead_normalized.view(batch_overhead, batch_surface, c, h, overhead_width)

    # l2 normalize surface embedding
    batch_surface, c, h, surface_width = surface_embed.shape
    surface_normalized = surface_embed.reshape(batch_surface, -1)
    surface_normalized = torch.div(surface_normalized, torch.linalg.norm(surface_normalized, ord=2, dim=-1).unsqueeze(-1))
    surface_normalized = surface_normalized.view(batch_surface, c, h, surface_width)

    # calculate L2 distance
    distance = 2*(1-torch.sum(overhead_normalized * surface_normalized.unsqueeze(0), (2, 3, 4))) # shape = [batch_overhead, batch_surface]

    return distance


def triplet_loss(distances, alpha=10.):

    batch_size = distances.shape[0]

    matching_dists = torch.diagonal(distances)
    dist_surface2overhead = matching_dists - distances
    dist_overhead2surface = matching_dists.unsqueeze(1) - distances

    loss_surface2overhead = torch.sum(torch.log(1. + torch.exp(alpha * dist_surface2overhead)))
    loss_overhead2surface = torch.sum(torch.log(1. + torch.exp(alpha * dist_overhead2surface)))

    soft_margin_triplet_loss = (loss_surface2overhead + loss_overhead2surface) / (2. * batch_size * (batch_size - 1))

    return soft_margin_triplet_loss


def train(dataset='cvusa', fov=360, val_quantity=1000, batch_size=64, num_workers=12, num_epochs=999999):

    csv_path = Globals.dataset_paths[dataset]['train']

    # Data modification and augmentation
    transform = torchvision.transforms.Compose([
        ResizeCVUSA(fov),
        ImageNormalization(),
        PolarTransform()
    ])

    # Source the training and validation data
    trainval_set = ImagePairDataset(dataset=dataset, csv_path=csv_path, transform=transform)
    train_set, val_set = torch.utils.data.random_split(trainval_set, [len(trainval_set) -  val_quantity, val_quantity])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Neural networks
    surface_encoder = FOV_DSM(circ_padding=False).to(device)
    overhead_encoder = FOV_DSM(circ_padding=True).to(device)
    # if torch.cuda.device_count() > 1:
    #     surface_encoder = torch.nn.DataParallel(surface_encoder)
    #     overhead_encoder = torch.nn.DataParallel(overhead_encoder)

    # Loss function
    loss_func = triplet_loss

    # Optimizer
    all_params = list(surface_encoder.parameters()) \
                 + list(overhead_encoder.parameters())
    optimizer = torch.optim.Adam(all_params, lr=1.E-5)

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
                #overhead = data['overhead'].to(device)
                overhead = data['polar'].to(device)

                with torch.set_grad_enabled(phase == 'train'):

                    # Forward and loss (train and val)
                    surface_embed = surface_encoder(surface)
                    overhead_embed = overhead_encoder(overhead)

                    orientation_estimate = correlation(overhead_embed, surface_embed)
                    overhead_cropped = crop_overhead(overhead_embed, orientation_estimate, surface_embed.shape[3])

                    distance = l2_distance(overhead_cropped, surface_embed)

                    loss = loss_func(distance)

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
            torch.save(surface_encoder.state_dict(), './fov_{}_surface_best.pth'.format(int(fov)))
            torch.save(overhead_encoder.state_dict(), './fov_{}_overhead_best.pth'.format(int(fov)))


def test(dataset='cvusa', fov=360, batch_size=64, num_workers=16):

    csv_path = Globals.dataset_paths[dataset]['test']

    # Specify transformation, if any
    transform = torchvision.transforms.Compose([
        ResizeCVUSA(fov),
        ImageNormalization(),
        PolarTransform()
    ])

    # Source the test data
    test_set = ImagePairDataset(dataset=dataset, csv_path=csv_path, transform=transform)
    #test_loader = torch.utils.data.DataLoader(test_set,sampler=torch.utils.data.SubsetRandomSampler(range(2000)), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Load the neural networks
    surface_encoder = FOV_DSM(circ_padding=False).to(device)
    overhead_encoder = FOV_DSM(circ_padding=True).to(device)
    surface_encoder.load_state_dict(torch.load('./fov_{}_surface_best.pth'.format(int(fov))))
    overhead_encoder.load_state_dict(torch.load('./fov_{}_overhead_best.pth'.format(int(fov))))
    surface_encoder.eval()
    overhead_encoder.eval()

    # Loop through batches of data
    surface_embed = None
    overhead_embed = None
    for batch, data in enumerate(tqdm.tqdm(test_loader)):
        surface = data['surface'].to(device)
        overhead = data['polar'].to(device)

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
        orientation_estimate = correlation(overhead_embed, this_surface_embed)
        overhead_cropped_all = crop_overhead(overhead_embed, orientation_estimate, this_surface_embed.shape[3])
        distances = l2_distance(overhead_cropped_all, this_surface_embed)
        distances = torch.squeeze(distances)
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


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
    parser.add_argument('--fov',
                        type=int,
                        default=360,
                        choices=range(6, 361),
                        metavar='{6-360}',
                        help='The field of view for cropping street level images. [Default = 360]')
    args = parser.parse_args()
    print(args)
    if args.mode == 'train':
        train(dataset=args.dataset, fov=args.fov)
    elif args.mode == 'test':
        test(dataset=args.dataset, fov=args.fov)
