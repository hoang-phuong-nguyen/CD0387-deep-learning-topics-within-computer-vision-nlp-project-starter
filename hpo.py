#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import argparse
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def test(model, test_loader, criterion, device):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item() 
        
    test_loss /= len(test_loader.dataset)
    logger.info(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )    


def train(model, train_loader, criterion, optimizer, num_epochs, device):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        total_correct = 0
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            pred = model(data)
            loss = criterion(pred, target)
            total_loss += loss
            loss.backward()
            optimizer.step()
            
            pred = pred.argmax(dim=1, keepdim=True)
            total_correct += pred.eq(target.view_as(pred)).sum().item()
        logger.info(f"Epoch {epoch}: Loss {total_loss/len(train_loader.dataset)}, Accuracy {100*(total_correct/len(train_loader.dataset))}%")
            
    return model 

    
def net(num_class):
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet50(pretrained=True)
    
    for param in model.parameters():
        param.required_grad = False 
    
    num_features = model.fc.in_features
    # model.fc = nn.Sequential(
    #     nn.Linear(num_features, 256), 
    #     nn.ReLU(),                 
    #     nn.Linear(256, 128),
    #     nn.ReLU(),
    #     nn.Linear(128, num_classes),
    #     nn.LogSoftmax(dim=1))
    model.fc = nn.Sequential(nn.Linear(num_features, num_class))
    return model


def create_data_loaders(data, batch_size, test_batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    train_path = os.path.join(data, 'train')
    test_path = os.path.join(data, 'test')
    val_path = os.path.join(data, 'valid')
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224))
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
        ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
        ])
    
    train_data = torchvision.datasets.ImageFolder(root=train_path, transform=train_transform)    
    val_data = torchvision.datasets.ImageFolder(root=val_path, transform=test_transform)
    test_data = torchvision.datasets.ImageFolder(root=test_path, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data, test_batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = net(args.num_class)
    model = model.to(device)
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    train_loader, _, test_loader = create_data_loaders(
        data=args.data_dir, 
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size
    )
    model = train(model, train_loader, loss_criterion, optimizer, args.epochs, device)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, test_loader, loss_criterion, device)
    
    '''
    TODO: Save the trained model
    '''
    torch.save(model.state_dict(), args.model_path)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify any training args that you might need
    '''
    parser=argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        metavar="N",
        help="number of epochs to train (default: 20)",
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=0.001, 
        metavar="LR", 
        help="learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--data-path", 
        type=str, 
        default="./dogImages", 
        help="path to input dataset"
    )
    parser.add_argument(
        "--num-class", 
        type=int, 
        default=133, 
        help="number of object classes"
    )    
    parser.add_argument(
        "--model-path", 
        type=str, 
        default="./model/best.pth", 
        help="path to trained model"
    )
    args = parser.parse_args()    
    main(args)