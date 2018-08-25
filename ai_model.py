#######################################
'''Functions and Classes relating to the Deep Learning Model'''
#######################################
from collections import OrderedDict
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms, models


def build_network(arch, lr, hidden_units): ################################################################
    # Pre-trained model
    if arch == 'vgg11':
        model = models.vgg11(pretrained=True)
        print("Model: vgg11")
    elif arch == 'alexnet':
        model = models.alexnet(pretrained = True)
        print("Model: alexnet")
    else:
        print("Only vgg11 and alexnet are valid to run this application. The default model, vgg11 is running now.")
        model = models.vgg11(pretrained=True)
    # Freeze parameters
    for param in model.parameters(): 
        param.requires_grad = False

    num_features = model.classifier[0].in_features


    # Build a feed-forward network
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(num_features, hidden_units)),
                                            ('relu', nn.ReLU()),
                                            ('dropout', nn.Dropout(p=0.5)),
                                            ('hidden', nn.Linear(hidden_units, 100)),
                                            ('relu', nn.ReLU()),
                                            ('dropout', nn.Dropout(p=0.5)),
                                            ('fc2', nn.Linear(100, 102)),
                                            ('output', nn.LogSoftmax(dim=1))]))
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    # Change to cuda
    model.to('cuda')
    
    return model, criterion, optimizer

# Function for the validation pass
def validation(model, validloader, criterion, optimizer): ########################################
    valid_loss = 0
    accuracy = 0
    for ii, (inputs, labels) in enumerate(validloader):
        inputs, labels = inputs.to('cuda:0'), labels.to('cuda:0')
        model.to('cuda:0')
        optimizer.zero_grad()
        with torch.no_grad():
            outputs = model.forward(inputs)
            valid_loss = criterion(outputs, labels)
            ps = torch.exp(outputs).data
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()
    return valid_loss, accuracy, optimizer


def train_network(model, trainloader, validloader, epochs, print_every, criterion, optimizer, power):############
    epochs = epochs
    print_every = 5
    steps = 0
    
    print("Training now.")
    for e in range(epochs):
        running_loss = 0
        
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            if power=='gpu':
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
                model.to('cuda')
            optimizer.zero_grad()
            # Forward and Backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if steps % print_every == 0:
                model.eval()
            
            valid_loss, accuracy, optimizer = validation(model, validloader, criterion, optimizer)
            print("Epoch:{}/{}...".format(e+1, epochs), 
                  "Training Loss: {:.4f}".format(running_loss/print_every), 
                  "Valid Loss: {:.3f}..".format(valid_loss/len(validloader)), 
                  "Accuracy: {:.3f}".format(accuracy/len(validloader)))
            running_loss = 0
    
    print("Finished Training.")
    return model
    
def save_checkpoint(filepath, arch, epochs, hidden_units, train_data, optimizer, model):#########################
    # Save the checkpoint 
    model.class_to_idx = train_data.class_to_idx

    checkpoint = {
        'arch': arch,
        'epoch': epochs,
        'hidden_units': hidden_units,
        'optimizer': optimizer.state_dict,
        'class_to_idx': model.class_to_idx, 
        'state_dict': model.state_dict()
    }

    torch.save(checkpoint, filepath)
    print("checkpoint.pth is saved.")
    
# Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):###############################################################
    checkpoint = torch.load(filepath)
    if checkpoint['arch'] == 'alexnet':
        model = models.alexnet(pretrained = True)
    else:
        model = models.vgg11(pretrained =True)
        
    arch = checkpoint['arch']
    #epoch = checkpoint['epoch']
    hidden_units = checkpoint['hidden_units']
    
    num_features = model.classifier[0].in_features
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(num_features, hidden_units)),
                                        ('relu', nn.ReLU()),
                                        ('dropout', nn.Dropout(p=0.5)),
                                        ('hidden', nn.Linear(hidden_units, 100)),
                                        ('relu', nn.ReLU()),
                                        ('dropout', nn.Dropout(p=0.5)),
                                        ('fc2', nn.Linear(100, 102)),
                                        ('output', nn.LogSoftmax(dim=1))]))
    model.classifier = classifier
    model.optimizer = checkpoint['optimizer']
    class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model, class_to_idx


def predict(img_torch, model, topk, power): ##################################################
    ''' 
    Predict the class (or classes) of an image using a trained deep learning model.
    '''
    if power == 'gpu':
        model.to('cuda')
        print('Predicting with GPU')
    else:
        model.cpu()
        print('Predicting with CPU')
    #img_torch = process_image(img_path)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()
    
    with torch.no_grad():
        output = model.forward(img_torch.cuda())
        
    prob = F.softmax(output.data,dim=1)
    
    print("Finished running the predict fuction")
    return prob.topk(topk)
    
    