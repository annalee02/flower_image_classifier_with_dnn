from collections import OrderedDict
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import argparse

import utility # utility.py
import ai_model # ai_model.py

parser = argparse.ArgumentParser(description='train.py')

# Define command line arguments

parser.add_argument('data_dir', nargs='*', action="store", default="./flowers/")
parser.add_argument('--save_dir', default="./checkpoint.pth", dest="save_dir", help='Set directory to save checkpoints')
parser.add_argument('--arch', default="vgg11", dest="arch", type=str, help='Choose architecture')
parser.add_argument('--lr', default="0.001", dest="lr", type=float, help='Set Learning rate')
parser.add_argument('--hidden_units', default="512", dest="hidden_units", type=int, help='Set hyperparameters')
parser.add_argument('--epochs', default = "2", dest="epochs", type=int, help='Set hyperparameters')
parser.add_argument('--gpu', default='gpu', dest='gpu', help='Use GPU for inference')

args = parser.parse_args()
data_dir = args.data_dir
filepath = args.save_dir
arch = args.arch
lr = args.lr
hidden_units = args.hidden_units
epochs = args.epochs
power = args.gpu

trainloader, validloader, testloader, train_data = utility.load_data()

model, criterion, optimizer = ai_model.build_network(arch, lr, hidden_units)
valid_loss, accuracy, optimizer = ai_model.validation(model, validloader, criterion, optimizer)

# Prints out training loss, validation loss, and validation accuracy as the network trains
ai_model.train_network(model, trainloader, validloader, epochs, 5, criterion, optimizer, power)

ai_model.save_checkpoint(filepath, arch, epochs, hidden_units, train_data, optimizer, model)

print("---End---")



