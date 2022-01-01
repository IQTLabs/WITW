import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import os
from PIL import Image
from tifffile import imread, imwrite
import tqdm

from cvig_fov import *
import pytorch_zoo


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dataset='cvusa'
fov=360

class Globals:
    surface_height_max = 128
    surface_width_max = 512
    overhead_size = 736

    dataset_paths = {
        'cvusa': {
            'train': '../data/train-19zl.csv',
            'test': '../data/val-19zl.csv'
        },
        'witw': {
            'train':'../dataset_v02/train_scenes.csv',
            'test':'../dataset_v02/test_scenes.csv'
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



csv_path = Globals.dataset_paths[dataset]['train']

# Data modification and augmentation
transform = torchvision.transforms.Compose([
    Resize(fov, dataset=='cvusa')
])

# Source the dataset
train_augment = ImagePairDataset(dataset=dataset, csv_path=csv_path, transform=transform)

# load cresi weights 
model = torch.load('../fold0_best.pth').to(device)

# create cresi directory to store tif files 
dir_name = 'cresi_uint8'
if not os.path.exists(os.path.join(os.path.dirname(csv_path), dir_name)):
    os.mkdir(os.path.join(os.path.dirname(csv_path), dir_name))
    print('created dir = ',os.path.join(os.path.dirname(csv_path), dir_name))

for i in tqdm.tqdm(range(len(train_augment))):

    file_paths_idx = train_augment[i]['idx']
    #overhead_raw = io.imread(train_augment.file_paths.iloc[file_paths_idx]['overhead'])
    pytorch_image = (train_augment[i]['overhead']/255.).unsqueeze(0).to(device)

    road_prediction = torch.sigmoid(model(pytorch_image)).squeeze()
    pytorch_image = torch.moveaxis(pytorch_image.squeeze(),0, -1 )
    road_prediction = torch.moveaxis(road_prediction, 0, -1)
    # min max normalize road prediction
    road_prediction = (road_prediction - torch.min(road_prediction))/ (torch.max(road_prediction) - torch.min(road_prediction))
    # scale road_prediction to 255
    road_prediction *= 255
    # scale pytorch_image back to 255
    pytorch_image *= 255

    combined = torch.cat((pytorch_image, road_prediction), 2).type(torch.uint8).detach().cpu().numpy()

    imwrite(os.path.join(os.path.dirname(csv_path), dir_name, os.path.splitext(os.path.basename(train_augment.file_paths.iloc[file_paths_idx]['overhead']))[0]+'.tif'), combined, planarconfig='CONTIG')

