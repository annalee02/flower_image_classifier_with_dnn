#######################################
'''Utility functions: Loading data and Preprocessing images'''
#######################################
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image

def load_data():#######################################################################
    # Directories
    data_dir = './flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Define the transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomSizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])


    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform = valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform = test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    
    return trainloader, validloader, testloader, train_data

def process_image(img_path):##############################################################
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    pil_image = Image.open(img_path)
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Process a PIL image for use in a PyTorch model
    image_transforms = transforms.Compose([transforms.Resize(256), # 256 pixels
                                           transforms.CenterCrop(224), # Crop out the center224x224
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean, std)])
    
    pil_image = image_transforms(pil_image)

    return pil_image
