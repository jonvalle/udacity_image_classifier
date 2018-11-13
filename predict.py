'''
Predict flower name from an image with predict.py along with the probability of that name.
That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

    Basic usage: python predict.py /path/to/image checkpoint
    Options:
        Return top KKK most likely classes: python predict.py input checkpoint --top_k 3
        Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
        Use GPU for inference: python predict.py input checkpoint --gpu
'''

import torch
import numpy as np
import json
import argparse
import utils

# Main program function defined below
def main():
    print('Welcome aboard. Fasten your seatbelts')
    #read parameters
    arguments = read_args()

    processor = utils.get_processor(arguments.gpu)

    #load checkpoint
    checkpoint = load_checkpoint(utils.basepath+arguments.checkpoint, processor)

    #download pre-trained model and initialize
    model, criterion, optimizer = utils.create_model(
                                        checkpoint['arch'],
                                        checkpoint['hidden_size'],
                                        checkpoint['output_size'],
                                        checkpoint['learning_rate'],
                                        checkpoint['dropout_per'])

    #load trained model in initialized model
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])

    #load image and topk, then predict its classification
    flower_probs, flower_classes = predict(utils.basepath+arguments.input,
                                            model,
                                            arguments.top_k)

    #chew stats and print results of an image and plot
    cat_to_name = load_cat_names(utils.basepath+arguments.category_names)
    print_results(flower_probs,
                  flower_classes,
                  checkpoint['img_mapping'],
                  utils.basepath+arguments.input,
                  cat_to_name)


    print("**INFO: Prediction Ended!")


def read_args():
    # Creates Argument Parser object named parser
    parser = argparse.ArgumentParser()

    # Argument 1: path to load
    parser.add_argument('--checkpoint', type = str, required = True,
                        help = 'path to the saved trained model')

    # Argument 2: path to image file to check
    parser.add_argument('--input', type = str,  required = True, default = 'flowers/test/37/image_03741.jpg',
                        help = 'image path to be predicted')

    # Argument 3: top results
    parser.add_argument('--top_k', type = int, default = '5',
                        help = 'display top KKK most likely classes')

    # Argument 4: ids of folders with mapping to images - source of truth
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json',
                        help = 'mapping of categories to real names')

    # Argument 5: choose archietcture
    parser.add_argument('--gpu', action='store_true',
                        help = 'use gpu acceleration if available')

    # Assigns variable in_args to parse_args()
    in_args = parser.parse_args()
    print("Arguments: {} ", in_args)

    return in_args

def load_checkpoint(filepath, processor):
    if processor == 'cuda':
        checkpoint = torch.load(filepath)
    else:
        checkpoint = torch.load(filepath, map_location=processor)
    print('**INFO: Loaded checkpoint')

    return checkpoint

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    print('**INFO: Start image prediction of '+image_path)
    img = utils.process_image(image_path)
    img = img.unsqueeze(0)

    model.type(torch.DoubleTensor)
    model.eval()

    # Calculate the class probabilities (softmax) for img
    with torch.no_grad():
        probability = model.forward(img)

    probs, classes = torch.exp(probability).topk(topk)
    np_prob = np.asarray(probs)[0]
    np_classes = np.asarray(classes)[0]

    print('**INFO: Finished predition')
    return (np_prob, np_classes)

def print_results(flower_probs, flower_classes, img_mapping, img_path, categories_names):
    flower_names = []
    for i in range(len(flower_probs)):
        most_prob = str(round(flower_probs.item(i)*100, 4))
        most_prob_flower_id = flower_classes.item(i)
        most_prob_flower_id_folder = get_folder_key(img_mapping,most_prob_flower_id)
        most_prob_flower = get_class_value(categories_names,most_prob_flower_id_folder)
        flower_names.append(most_prob_flower)
        print("The image has a {:.2f}% probability that is a {}".format(float(most_prob),most_prob_flower))
    utils.display_image_and_chart(img_path, most_prob, flower_names)

def load_cat_names(cat_to_name):
    with open(cat_to_name, 'r') as f:
        categories = json.load(f)
        return categories

def get_folder_key(mydict,value):
    return list(mydict.keys())[list(mydict.values()).index(value)]

def get_class_value(categories_names, key):
    return categories_names[key]

# Call to main function to run the program
if __name__ == "__main__":
    main()
