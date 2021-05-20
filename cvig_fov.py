import torch
import torchvision
from torch.nn.modules.utils import  _reverse_repeat_tuple
import matplotlib.pyplot as plt
import math

from cvig import *


class ResizeCVUSA(object):
    """
    Resize the CVUSA images to fit model.
    """
    def __init__(self, fov):
        self.fov = fov
        self.surface_height = 128
        self.surface_width = int(self.fov / 360 * 512)
        self.surface_resize_width = 512
        self.overhead_height = 256
        self.overhead_width = 256

    def __call__(self, data):
        start = np.random.randint(0, self.surface_resize_width-self.surface_width+1)
        end = start + self.surface_width
        data['surface'] = torchvision.transforms.functional.resize(data['surface'], (self.surface_height, self.surface_resize_width))[:,:,start:end]

        data['overhead'] = torchvision.transforms.functional.resize(data['overhead'], (self.overhead_height, self.overhead_width))
        return data

class PolarTransform(object):
    """
    Applies polar transform from "Where am I looking at? Joint Location and Orientation Estimation by Cross-View Matching"
    CVPR 2020.
    """

    def __call__(self, data):
        assert data['overhead'].shape[1] == data['overhead'].shape[2]
        size_a = data['overhead'].shape[1]
        height_g = data['surface'].shape[1]
        width_g = data['surface'].shape[2]

        transf_aerial = torch.zeros(data['surface'].shape)

        for x in range(width_g):
            for y in range(height_g):
                y_a = (size_a/2) + ((size_a/2) * (height_g -1 - y)/height_g * math.cos(2 * math.pi * x / width_g) )
                x_a = (size_a/2) - ((size_a/2) * (height_g -1 - y)/height_g * math.sin(2 * math.pi * x / width_g) )
                transf_aerial[:,y,x] = data['overhead'][:,int(y_a),int(x_a)]

        data['polar'] = transf_aerial
        return data

def prep_model(circ_padding=False):
    """
    Prepare vgg16 model with modification from "Where am I looking at? Joint Location and Orientation Estimation by Cross-View Matching"
    CVPR 2020.
    """
    model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16_bn', pretrained=True)

    # shorten model based on Where Am I Looking modifications
    model.features = model.features[:23]

    model.features.add_module(str(len(model.features)), torch.nn.Conv2d(512, 256, 3, (2, 1), padding=1))
    model.features.add_module(str(len(model.features)), torch.nn.ReLU(inplace=True))
    model.features.add_module(str(len(model.features)), torch.nn.Conv2d(256, 64, 3, (2, 1), padding=1))
    model.features.add_module(str(len(model.features)), torch.nn.ReLU(inplace=True))
    model.features.add_module(str(len(model.features)), torch.nn.Conv2d(64, 16, 3, padding=1))

    # only train last 3 conv layers
    for name, param in model.features.named_parameters():
        torch_layer_num = int(name.split('.')[0])
        if torch_layer_num < 17:
            param.requires_grad = False

    # circular padding
    if circ_padding:
        for layer in model.features:
            if isinstance(layer, torch.nn.Conv2d):
                # TODO: circular padding horizontal, but zero padding vertical ?
                layer.padding = (1,2)
                layer.padding_mode = 'circular'
                layer._reversed_padding_repeated_twice = _reverse_repeat_tuple(layer.padding, 2)

    return model

def correlation(overhead_embed, surface_embed):

    o_c, o_h, o_w = overhead_embed.shape[1:]
    s_c, s_h, s_w = surface_embed.shape[1:]
    assert o_h == s_h, o_c == s_c

    # append beginning of overhead embedding to the end to get a full correlation
    n = s_w - 1
    x = torch.cat((overhead_embed, overhead_embed[:,:, :, :n]), axis=3)
    f = surface_embed

    # calculate correlation using convolution
    out = torch.nn.functional.conv2d(x, f,  stride=1)
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
    overhead_embed = torch.unsqueeze(overhead_embed, 1) # shape=[batch_overhead, 1, c, h, w]
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
    distance = 2*(1-torch.sum(overhead_normalized * surface_normalized.unsqueeze(0), (2, 3, 4))) # shape = [batch_surface, batch_overhead]

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

def train(csv_path = './data/train-19zl.csv', fov=360, val_quantity=1000, batch_size=64, num_workers=16, num_epochs=999999):

    # Data modification and augmentation
    transform = torchvision.transforms.Compose([
        #Reorient(), #QuantizedSyncedRotation(),
        #OrientationMaps(),
        #SurfaceVertStretch()
        ResizeCVUSA(fov),
        PolarTransform()
    ])

    # Source the training and validation data
    trainval_set = ImagePairDataset(csv_path=csv_path, transform=transform)
    train_set, val_set = torch.utils.data.random_split(trainval_set, [len(trainval_set) -  val_quantity, val_quantity])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Neural networks
    model = prep_model().to(device)
    model_circ = prep_model(circ_padding=True).to(device)

    surface_encoder = model
    overhead_encoder = model_circ

#     if torch.cuda.device_count() > 1:
#         surface_encoder = nn.DataParallel(surface_encoder)
#         overhead_encoder = nn.DataParallel(overhead_encoder)
    # Loss function
    loss_func = triplet_loss

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
                #overhead = data['overhead'].to(device)
                overhead = data['polar'].to(device)

                with torch.set_grad_enabled(phase == 'train'):

                    # Forward and loss (train and val)
                    surface_embed = surface_encoder.features(surface)
                    overhead_embed = overhead_encoder.features(overhead)

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

                print('iter = {}, count = {}, loss = {}, running loss = {}'.format(batch, running_count, loss, running_loss))

            print('  %5s: avg loss = %f' % (phase, running_loss / running_count))

        # Save weights if this is the lowest observed validation loss
        if best_loss is None or running_loss / running_count < best_loss:
            print('-------> new best')
            best_loss = running_loss / running_count
            torch.save(surface_encoder.state_dict(), './fov_{}_surface_best.pth'.format(int(fov)))
            torch.save(overhead_encoder.state_dict(), './fov_{}_overhead_best.pth'.format(int(fov)))

def test(csv_path = './data/val-19zl.csv', fov=360, batch_size=12, num_workers=8):

    # Specify transformation, if any
    transform = torchvision.transforms.Compose([
        ResizeCVUSA(fov),
        PolarTransform()
    ])


    # Source the test data
    test_set = ImagePairDataset(csv_path=csv_path, transform=transform)
    #test_loader = torch.utils.data.DataLoader(test_set,sampler=torch.utils.data.SubsetRandomSampler(range(2000)), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)


    # Load the neural network
    # Neural networks
    model = prep_model().to(device)
    model_circ = prep_model(circ_padding=True).to(device)

    surface_encoder = model
    overhead_encoder = model_circ


    surface_encoder.load_state_dict(torch.load('./fov_{}_surface_best.pth'.format(int(fov))))
    overhead_encoder.load_state_dict(torch.load('./fov_{}_overhead_best.pth'.format(int(fov))))
    surface_encoder.eval()
    overhead_encoder.eval()

    # Loop through batches of data
    surface_embed = None
    overhead_embed = None
    for batch, data in enumerate(test_loader):
        print('[{}/{}]'.format(batch, len(test_loader)))
        surface = data['surface'].to(device)
        overhead = data['polar'].to(device)

        with torch.set_grad_enabled(False):
            surface_embed_part = surface_encoder.features(surface)
            overhead_embed_part = overhead_encoder.features(overhead)

            if surface_embed is None:
                surface_embed = surface_embed_part
                overhead_embed = overhead_embed_part
            else:
                surface_embed = torch.cat((surface_embed, surface_embed_part), dim=0)
                overhead_embed = torch.cat((overhead_embed, overhead_embed_part), dim=0)

    # Measure performance
    count = surface_embed.size(0)
    ranks = np.zeros([count], dtype=int)
    for idx in range(count):
        this_surface_embed = torch.unsqueeze(surface_embed[idx, :], 0)
        overhead_cropped_all = None

        ###
        orientation_estimate = correlation(overhead_embed, this_surface_embed)
        overhead_cropped_all = crop_overhead(overhead_embed, orientation_estimate, this_surface_embed.shape[3])
        overhead_cropped_all = overhead_cropped_all.reshape(overhead_cropped_all.shape[0], -1)
        ###
        print('[{}/{}]'.format(idx, count))
        '''
        for idx2 in range(count):
            print('[{}/{}][{}/{}]'.format(idx, count, idx2, count))
            this_overhead_embed = torch.unsqueeze(overhead_embed[idx2, :], 0)
            #this_overhead_embed = overhead_embed[idx2, :]

            orientation_estimate = correlation(this_overhead_embed, this_surface_embed)
            overhead_cropped = crop_overhead(this_overhead_embed, orientation_estimate, this_surface_embed.shape[3])
            overhead_cropped = overhead_cropped.squeeze(0)
            if overhead_cropped_all is None:
                overhead_cropped_all = overhead_cropped
            else:
                overhead_cropped_all = torch.cat((overhead_cropped_all, overhead_cropped), dim=0)

        #overhead_cropped_all = overhead_cropped_all.reshape(overhead_cropped_all.shape[0], -1)
        ####
        '''
        this_surface_embed = this_surface_embed.reshape(this_surface_embed.shape[0], -1)
        distances = torch.pow(torch.sum(torch.pow(overhead_cropped_all - this_surface_embed, 2), dim=1), 0.5)
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
    fov = 360
    if len(sys.argv) == 3:
        fov = float(sys.argv[2])
    if len(sys.argv) < 2 or sys.argv[1]=='train':
        train(fov=fov)
    elif sys.argv[1]=='test':
        test(fov=fov)
