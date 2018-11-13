'''
BACKUP FROM SERVER BEFORE THE NEW SAVE CHECKPOINT FORMAT
Train a new network on a data set with train.py
    Basic usage: python train.py data_directory
    Prints out training loss, validation loss, and validation accuracy as the network trains
    Options:
        Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
        Choose architecture: python train.py data_dir --arch "vgg13"
        Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
        Use GPU for training: python train.py data_dir --gpu
'''

import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json
from PIL import Image
import argparse
from collections import OrderedDict
import utils

dropout_per = 0.25
n_classes = 102

# Main program function defined below
def main():
    print('Welcome aboard. Fasten your seatbelts')
    #read parameters
    arguments = read_args()

    #generate and normalize images
    train_dataset, train_classes_ids = utils.generate_dataset('train')

    #download pre-trained model and initialize
    model, criterion, optimizer = create_model(arguments.arch,
                                               arguments.hidden_units,
                                               n_classes,
                                               arguments.learning_rate,
                                               dropout_per)

    processor = utils.get_processor(arguments.gpu)

    #train network
    model, optimizer = do_deep_learning(model, train_dataset, arguments.epochs, 10, criterion, optimizer, processor)

    #test network with test dataset
    test_dataset, test_classes_ids = utils.generate_dataset('test')
    check_accuracy_on_test(test_dataset, model, processor)

    #load checkpoint
    #checkpoint = load_checkpoint(utils.basepath+utils.checkpoint, processor)
    #model.load_state_dict(checkpoint['model'])
    #optimizer.load_state_dict(checkpoint['optimizer'])

    #save checkpoint
    ''' OLD FORMAT
    save_params_dict = {'hidden_size': arguments.hidden_units,
                        'output_size': n_classes,
                        'dropout_per': dropout_per,
                      'learning_rate': arguments.learning_rate,
                     'epochs_trained': arguments.epochs,
                        'img_mapping': train_classes_ids,
                          'optimizer': optimizer.state_dict(),
                              'model': model.state_dict()}
    '''
    save_params_dict = {'hidden_size': arguments.hidden_units,
                        'output_size': n_classes,
                        'dropout_per': dropout_per,
                      'learning_rate': arguments.learning_rate,
                     'epochs_trained': arguments.epochs,
                        'img_mapping': train_classes_ids,
                    'optimizer_state': optimizer.state_dict(),
                        'model_state': model.state_dict(),
                         'classifier': model.classifier,
                               'arch': arguments.arch,
                         'input_size': model.classifier[0].in_features}    
    save_checkpoint(utils.basepath+'checkpoint_vgg13_2.pth',save_params_dict)

    #load checkpoint
    checkpoint = load_checkpoint(utils.basepath+'checkpoint_vgg13_2.pth', processor)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])

    print("**INFO: Training Ended!")

def read_args():
    # Creates Argument Parser object named parser
    parser = argparse.ArgumentParser()

    # Argument 1: path to save
    parser.add_argument('--save_dir', type = str, default = '',
                        help = 'path to save the trained model')

    # Argument 2: choose archietcture
    parser.add_argument('--arch', type = str, default = 'vgg16',
                        help = 'name of the chosen architecture')

    # Arguments 3-5: training hyperparameters
    parser.add_argument('--learning_rate', type = float, default = '0.01',
                        help = 'training: learning rate')

    parser.add_argument('--hidden_units', type = int, default = '1000',
                        help = 'training: hidden units')

    parser.add_argument('--epochs', type = int, default = '20',
                        help = 'training: number of epochs')

    # Argument 6: choose archietcture
    parser.add_argument('--gpu', type = bool, default = 'True',
                        help = 'use gpu acceleration if available')

    # Assigns variable in_args to parse_args()
    in_args = parser.parse_args()
    print("Arguments: {} ", in_args)

    return in_args



def create_model(model_type, hidden_size, output_size,
                    learning_rate, dropout_per=0.25):

    model = eval(utils.get_model(model_type))

    print("**INFO - Loading Model: {} - Hidden units: {} - LR: {}".format(model_type,hidden_size,learning_rate))
    for param in model.parameters():
        param.requires_grad = False

    input_size = model.classifier[0].in_features

    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_size, hidden_size)),
                              ('relu1', nn.ReLU()),
                              ('dropout', nn.Dropout(dropout_per)),
                              ('fc2', nn.Linear(hidden_size, output_size)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    return (model, criterion, optimizer)

def do_deep_learning(model, trainloader, epochs, print_every, criterion, optimizer, device):
    steps = 0
    print('**INFO: Start deep learning')
    
    # change to cuda/cpu
    model.to(device)
    model.train()

    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every))

                running_loss = 0

    print("**INFO: Finished Deep Learning")
    return (model, optimizer)

def check_accuracy_on_test(testloader, model, processor):
    correct = 0
    total = 0
    model.eval()
    print('**INFO: Checking accuracy of trained model')

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            if processor == 'cuda':
                images = images.cuda()
                labels = labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network with {} test images: {}'.format(total,(100 * correct / total)))

def save_checkpoint(filepath, checkpoint):
    res = torch.save(checkpoint, filepath)
    print('**INFO: Saved checkpoint as: {} - code: {}'.format(filepath,res))

def load_checkpoint(filepath, processor):
    if processor == 'cuda':
        checkpoint = torch.load(filepath)
    else:
        checkpoint = torch.load(filepath, map_location=processor)
    print('**INFO: Loaded checkpoint')
    return checkpoint



# Call to main function to run the program
if __name__ == "__main__":
    main()

