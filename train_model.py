#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import os 
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import argparse
import logging
import smdebug.pytorch as smd


#TODO: Import dependencies for Debugging andd Profiling
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def test(model, test_loader, criterion, hook):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
    hook.set_mode(smd.modes.EVAL)
    test_loss = 0
    running_corrects = 0
    
    for inputs, labels in test_loader:
        outputs = model(inputs)
        test_loss += criterion(outputs, labels).item()
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds==labels.data).item()
        
    average_accuracy = running_corrects / len(test_loader.dataset)
    average_loss = test_loss / len(test_loader.dataset)
    logger.info(f'Test set: Average loss: {average_loss}, Average accuracy: {100*average_accuracy}%')
    

def train(model, train_loader, validation_loader, criterion, optimizer, num_epochs=100):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)
    count = 0
    
    for epoch in range(num_epochs):
        hook.set_mode(smd.modes.TRAIN)
        model.train()
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            count += len(inputs)
            
        hook.set_mode(smd.modes.EVAL)
        model.eval()
        running_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in validation_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data).item()
                
        total_accuracy = running_corrects / len(validation_loader.dataset)
        logger.info(f'Validation set: Average accuracy: {100*total_accuracy}%')
    return model    
    
    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet50(pretrained = True)
    
    for param in model.parameters():
        param.required_grad = False
    
    num_features = model.fc.in_features
    num_classes = 133
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256), 
        nn.ReLU(),                 
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128,  num_classes),
        nn.LogSoftmax(dim=1))
    return model


def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    train_path = os.path.join(data, 'train')
    test_path = os.path.join(data, 'test')
    validation_path = os.path.join(data, 'valid')
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize(256),
        transforms.Resize((224, 224)),
        transforms.ToTensor()])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.Resize((224, 224)),
        transforms.ToTensor()])
    
    train_dataset = torchvision.datasets.ImageFolder(root=train_path, transform=train_transform)    
    test_dataset = torchvision.datasets.ImageFolder(root=test_path, transform=test_transform)
    validation_dataset = torchvision.datasets.ImageFolder(root=validation_path, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, validation_loader


def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    logger.info("Initialize a model by calling the net function")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = net()
    model = model.to(device)
    
    '''
    TODO: Create your loss and optimizer
    '''
    logger.info("Create loss and optimizer")
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    logger.info("Start training the model")
    train_loader, test_loader, validation_loader = create_data_loaders(
        data=args.data_dir, 
        batch_size=args.batch_size)
    
    model = train(model, train_loader, validation_loader, loss_criterion, optimizer)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    logger.info("Test the model to see its accuracy")
    test(model, test_loader, loss_criterion)
    
    '''
    TODO: Save the trained model
    '''
    logger.info("Save the trained model")
    torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pth'))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify any training args that you might need
    '''
    parser=argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64, metavar="N", help="input batch size for training")
    parser.add_argument( "--test_batch_size", type=int, default=1000, metavar="N", help="input batch size for testing")
    parser.add_argument("--epochs", type=int, default=2, metavar="N", help="number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.01, metavar="LR", help="learning rate")
    parser.add_argument("--data_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"], help="training data path in S3")
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"], help="location to save the model to")
    args=parser.parse_args()
    
    main(args)
