'''
Train a new network on a data set with train.py
    Basic usage: python train.py data_directory
    Prints out training loss, validation loss, and validation accuracy as the network trains
    Options:
        Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
        Choose architecture: python train.py data_dir --arch "vgg13"
        Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
        Use GPU for training: python train.py data_dir --gpu
        Example: python train.py --gpu --data_dir flowers --arch vgg13 --epochs 1
'''

import torch
import argparse
import utils

dropout_per = 0.25
n_classes = 102
checkpoint_name = 'checkpoint.pth'

# Main program function defined below
def main():
    print('Welcome aboard. Fasten your seatbelts')
    #read parameters
    arguments = read_args()

    #generate and normalize images
    train_dataset, train_classes_ids = utils.generate_dataset('train', arguments.data_dir)

    #download pre-trained model and initialize
    model, criterion, optimizer = utils.create_model(arguments.arch,
                                               arguments.hidden_units,
                                               n_classes,
                                               arguments.learning_rate,
                                               dropout_per)

    processor = utils.get_processor(arguments.gpu)

    #train network
    model, optimizer = do_deep_learning(model, 
                                        train_dataset, 
                                        arguments.epochs, 
                                        10, #print every
                                        criterion, 
                                        optimizer, 
                                        processor)

    #test network with test dataset
    test_dataset, test_classes_ids = utils.generate_dataset('test', arguments.data_dir)
    check_accuracy_on_test(test_dataset,
                           model,
                           processor)

    #save checkpoint
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

    save_checkpoint(utils.basepath+arguments.save_dir+checkpoint_name,save_params_dict)

    print("**INFO: Training Ended!")


def read_args():
    # Creates Argument Parser object named parser
    parser = argparse.ArgumentParser()

    # Argument 0: data directory (flowers?)
    parser.add_argument('--data_dir', type = str, required = True,
                        help = 'name of the subfolder that contains the datasets')
    
    # Argument 1: path to save
    parser.add_argument('--save_dir', type = str, default = './',
                        help = 'path to save the trained model')

    # Argument 2: choose archietcture
    parser.add_argument('--arch', type = str, default = 'vgg16',
                        help = 'name of the chosen architecture')

    # Arguments 3-5: training hyperparameters
    parser.add_argument('--learning_rate', type = float, default = '0.001',
                        help = 'training: learning rate')

    parser.add_argument('--hidden_units', type = int, default = '1000',
                        help = 'training: hidden units')

    parser.add_argument('--epochs', type = int, default = '5',
                        help = 'training: number of epochs')

    # Argument 6: choose archietcture
    parser.add_argument('--gpu', action='store_true',
                        help = 'use gpu acceleration if available')

    # Assigns variable in_args to parse_args()
    in_args = parser.parse_args()
    print("Arguments: {} ", in_args)

    return in_args


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


# Call to main function to run the program
if __name__ == "__main__":
    main()
