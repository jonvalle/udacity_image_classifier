'''
BACKUP BEFORE SAVE CHECKPOTIN CHANGES AND MOVING THE MODEL CREATION TO UTILS
'''

import json
from PIL import Image
import numpy as np
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

basepath = './'
checkpoint = 'checkpoint_sunflower.pth'
supp_models = {'resnet18': 'models.resnet18(pretrained=True)',
                  'vgg16': 'models.vgg16(pretrained=True)',
                  'vgg13': 'models.vgg13(pretrained=True)',
                'alexnet': 'models.alexnet(pretrained=True)'}


with open(basepath+'cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

def get_folder_key(mydict,value):
    return list(mydict.keys())[list(mydict.values()).index(value)]

def get_class_value(key):
    return cat_to_name[key]

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array *** ERROR: SHOULD RETURN A TENSOR ***
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(image)
    img = img.resize((256,256))
    img = img.crop((0,0,224,224))

    means = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # TODO: convert color to floats 0-1, normalize, etc
    np_image = np.array(img)
    np_image_float = np.array(img)/255
    np_image_normalized = (np_image_float-means)/std
    np_image_transposed = np_image_normalized.transpose((2,0,1))

    return torch.from_numpy(np_image_transposed)

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax

def generate_dataset(data_type):
    path = basepath
    if (data_type == 'train'):
        path += 'flowers/train'
    elif (data_type == 'valid'):
        path += 'flowers/valid'
    elif (data_type == 'test'):
        path += 'flowers/train'
    else:
        #some error
        return None

    if (data_type == 'train'):
        data = transforms.Compose([transforms.RandomRotation(30),
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225])])
    else:
        data = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225])])

    image_datasets = datasets.ImageFolder(path, transform=data)

    if (data_type == 'train'):
        dataloader = torch.utils.data.DataLoader(image_datasets, batch_size=64, shuffle=True)
    else:
        dataloader = torch.utils.data.DataLoader(image_datasets, batch_size=32)

    return (dataloader, image_datasets.class_to_idx)

def get_model(id):
    if supp_models.get(id) is not None:
        print("**Loading Model {}".format(id))
        return supp_models[id]
    else:
        default = 'vgg16'
        print("**Warning: Model {} is not available. Defaulting to {}".format(id, default))
        return supp_models[default]

def get_processor(arg):
    if torch.cuda.is_available() and arg == True:
        print("**INFO: using GPU/Cuda")
        return 'cuda'
    else:
        print("**INFO: using CPU")
        return 'cpu'

