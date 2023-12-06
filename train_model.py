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

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

#TODO: Import dependencies for Debugging andd Profiling
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def test(model, test_loader, criterion, hook, device):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
    hook.set_mode(smd.modes.EVAL)
    loss = 0.0
    correct = 0.0
    
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        _, pred = torch.max(output, 1)
        loss += float(criterion(output, target).item() * data.size(0))
        correct += float(torch.sum(pred==target.data))

    average_loss = float(loss) // float(len(test_loader.dataset))
    average_accuracy = float(correct) // float(len(test_loader.dataset))
    
    logger.info(f"Average loss: {average_loss}")
    logger.info(f"Average accuracy: {average_accuracy}")


def train(model, train_loader, validation_loader, criterion, optimizer, epochs, hook, device):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''    
    image_dataset = {'train':train_loader, 'valid':validation_loader}
    best_loss = float(1e6)
    
    for epoch in range(epochs):
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
                hook.set_mode(smd.modes.TRAIN)
            else:
                model.eval()
                hook.set_mode(smd.modes.EVAL)
            running_loss = 0.0
            running_corrects = 0.0
            running_samples = 0
            
            total_samples_in_phase = len(image_dataset[phase].dataset)

            for inputs, labels in image_dataset[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)                  
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += float(loss.item() * inputs.size(0))
                running_corrects += float(torch.sum(preds==labels.data))
                running_samples +=len(inputs)

                accuracy = float(running_corrects)/float(running_samples)
                logger.info("Epoch {}, Phase {}, Images [{}/{} ({:.0f}%)] Loss: {:.2f} Accuracy: {}/{} ({:.2f}%)".format(
                    epoch,
                    phase,
                    running_samples,
                    total_samples_in_phase,
                    100.0 * (float(running_samples) / float(total_samples_in_phase)),
                    loss.item(),
                    running_corrects,
                    running_samples,
                    100.0*accuracy,
                ))
                 
                if (running_samples > (0.1*total_samples_in_phase)):
                    break
                
            current_loss = float(float(running_loss) / float(running_samples))
            current_acc = float(float(running_corrects) / float(running_samples))
            
            if phase == 'valid':
                if current_loss < best_loss:
                    best_loss = current_loss

            logger.info('{} loss: {:.4f}, acc: {:.4f}, best loss: {:.4f}'.format(phase, current_loss, current_acc, best_loss)) 
                    
    return model
    
    
def net(num_class):
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet50(pretrained=True, progress=True)

    for param in model.parameters():
        param.requires_grad = False   
    
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, num_class)
    )
    return model


def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    train_data_path = os.path.join(data, 'train')
    test_data_path = os.path.join(data, 'test')
    validation_data_path=os.path.join(data, 'valid')

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        ])

    train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=train_transform)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=test_transform)
    test_data_loader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    validation_data = torchvision.datasets.ImageFolder(root=validation_data_path, transform=test_transform)
    validation_data_loader  = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, shuffle=True) 
    
    return train_data_loader, test_data_loader, validation_data_loader


def save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)

    
def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = net(args.num_class)
    model.to(device)

    # create a SMDebug hook and register to the model 
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)
    
    '''
    TODO: Create your loss and optimizer
    '''
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    train_loader, test_loader, validation_loader = create_data_loaders(args.data_path, args.batch_size)
    model = train(model, train_loader, validation_loader, criterion, optimizer, args.epochs, hook, device)
   
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, test_loader, criterion, hook, device)
    
    '''
    TODO: Save the trained model
    '''
    save_model(model, args.model_dir)


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
#     parser.add_argument(
#         "--test-batch-size",
#         type=int,
#         default=32,
#         metavar="N",
#         help="input batch size for testing (default: 1000)",
#     )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=0.001, 
        metavar="LR", 
        help="learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--num-class", 
        type=int, 
        default=133, 
        help="number of object classes (default: 133)"
    )
    parser.add_argument('--data_path', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    args = parser.parse_args()
    
    main(args)
