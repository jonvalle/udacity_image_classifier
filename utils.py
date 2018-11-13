from PIL import Image
import numpy as np
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from collections import OrderedDict

basepath = '../Image_Classifier/'
#basepath = './'
supp_models = {'resnet18': 'models.resnet18(pretrained=True)',
                  'vgg16': 'models.vgg16(pretrained=True)',
                  'vgg13': 'models.vgg13(pretrained=True)',
                'alexnet': 'models.alexnet(pretrained=True)'}

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

def generate_dataset(data_type, data_dir):
    basepath = data_dir
    if (data_type == 'train'):
        basepath += '/train'
    elif (data_type == 'valid'):
        basepath += '/valid'
    elif (data_type == 'test'):
        basepath += '/test'
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

    image_datasets = datasets.ImageFolder(basepath, transform=data)

    if (data_type == 'train'):
        dataloader = torch.utils.data.DataLoader(image_datasets, batch_size=64, shuffle=True)
    else:
        dataloader = torch.utils.data.DataLoader(image_datasets, batch_size=32)

    return (dataloader, image_datasets.class_to_idx)

def get_model(id=None):
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

def create_model(model_type, hidden_size, output_size, learning_rate, dropout_per):

    model = eval(get_model(model_type))

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

def display_image_and_chart(img, probs, labels):
    ''' Function for viewing an image and it's predicted classes.
    '''
    #data = data.numpy().squeeze()

    #display 2 tables, same horizontal line
    fig, (ax1, ax2) = plt.subplots(figsize=(4,6), nrows=2)
    y_pos = np.arange(len(probs))

    #print image
    image = process_image(img)
    image = image.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    ax1.imshow(image)
    ax1.set_title(labels[0].title())
    ax1.axis('off')

    #draw chart
    ax2.barh(y_pos, probs, align='center',color='blue')
    ax2.set_aspect(0.1)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels)
    ax2.set_title('Flower Probability')
    ax2.set_xlim(0, 1.1)
    ax2.invert_yaxis()  # labels read top-to-bottom
    ax2.set_xlabel('Probability')

    plt.tight_layout()

    plt.show()
    print('Chart printed')
