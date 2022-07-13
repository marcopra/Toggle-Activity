import copy
import random
import torch

from models.AutoEncoder import AutoEncoder
from models.Decoder import ResnetDecoder
from utils import load_CIFAR10_datasets
from models.Resnet import resnet20
from models.utils import load_from_dict
from ToggleActivity.ToggleActivity import ToggleActivity

from torchinfo import summary

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Set a seed for reproducibility
seed = 1234
torch.manual_seed(seed)
random.seed(seed)

def fun():
    from matplotlib import pyplot as plt

    im = list(val_loader)[0][0]

    plt.imshow(im[0].permute(1, 2, 0))
    plt.show()

    pred = autoencoder(im)

    plt.imshow(pred.detach()[0].permute(1, 2, 0))
    plt.show()

# Specify whether to use cpu or gpu (based on gpu availability)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Test Dataset
train_loader, val_loader, test_loader = load_CIFAR10_datasets()

# Initialize the network and load the weights learned during training
network = resnet20()
load_from_dict(network, f"models/pretrained_models/resnet20.th", device)
network.eval()

# Initialize the autoencoder
encoder = copy.deepcopy(network)
load_from_dict(encoder, f"models/pretrained_models/resnet20.th", device)
encoder = torch.nn.Sequential(*(list(encoder.children())[:-1]))
decoder = ResnetDecoder()

autoencoder = AutoEncoder(encoder, decoder)
load_from_dict(autoencoder, 'models/weights/autoencoder_resnet20.pt', device)
# autoencoder.encoder = encoder

def invert_hook(model, input, output):
    return -output
encoder_summary = summary(autoencoder.decoder, verbose=False).summary_list
encoder_layer_list = [layer.module for layer in encoder_summary if layer.is_leaf_layer]
# for layer in encoder_layer_list:
#     layer.register_forward_hook(invert_hook)

dummy_lambda = lambda x: 1000 * torch.ones(size=x.shape)

toggle_activity = ToggleActivity(golden_network=network,
                                 toggle_network=copy.deepcopy(network),
                                 golden_dataloader=test_loader,
                                 toggle_dataset_function=autoencoder)

toggle_dataframe = toggle_activity.compute_toggle_activity(save_detailed_results=True, save_summarized_results=False)
toggle_dataframe.to_csv('results/resnet20/toggle_autoencoder_dataframe.csv', index=False)
