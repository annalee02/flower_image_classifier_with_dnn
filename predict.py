from collections import OrderedDict
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import argparse
import numpy as np

import utility # utility.py
import ai_model # ai_model.py

# Command Line Arguments
parser = argparse.ArgumentParser(description='predict.py')
parser.add_argument('--topk', default = 3, dest='topk', help='Return top K most likely classes', type=int)
parser.add_argument('--category_names', default = 'cat_to_name.json', dest='flower_names', help='Use a mapping of categories to real names')
parser.add_argument('--gpu', default='gpu', dest='gpu', help='Use GPU for inference')
parser.add_argument('checkpoint', default='/home/workspace/aipnd-project/checkpoint.pth', nargs='*', type = str)
parser.add_argument('input', default='./flowers/test/5/image_05159.jpg', nargs='*', type = str)

args = parser.parse_args()
img_path = args.input
topk = args.topk
power = args.gpu
filepath = args.checkpoint

trainloader, validloader, testloader, train_data = utility.load_data()
model = ai_model.load_checkpoint(filepath)
         
# Label mapping
import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
  
pil_image = utility.process_image(img_path)

result = ai_model.predict(pil_image, model, topk, power)
probs = F.softmax(result[0].data, dim=1).cpu().numpy()[0]
classes = result[1].data.cpu().numpy()[0]

    
print(probs)
print(classes)

labels = [cat_to_name[str(i)] for i in classes]


i = 0
while i < topk:
    print ("It's {} percent likely that the flower is {}.".format(probs[i], labels[i]))
    i += 1

print("---End---")

