import numpy as np
import torch
import torchvision
import json

from torchvision import transforms, datasets, models
from PIL import Image


pre_models = {"vgg16": 25088, "densenet121": 1024}


def load_image_names(filename):
    """Function that loads names of images to be trained
    Arg:
        filename: path of the file containing images
    Returns:
        images_names: dictionary of the images names
    """
    # filename must be a json file
    with open(filename) as f:
        image_names = json.load(f)

    return image_names


def load_data(dir_path):
    """ Function that loads the dataset(all images)

    Arg:
        dir_path: directory of folder containing images

    Returns:
        trainloader: loaded images for training
        testloader: loaded images for testing
        validloader: loaded images for validation
    """
    data_dir = dir_path
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Transforms needed to properly format images
    train_transforms = transforms.Compose([transforms.RandomRotation(20),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                           ])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                          ])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                           ])

    # Load the datasets with ImageFolder
    train_set = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_set = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_set = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_set, batch_size=64, shuffle=True)

    return trainloader, testloader, validloader, train_set


def save_checkpoint(path, optimizer, structure, model, lr, epochs, train_set):
    """ Function to save trained model

    Args:
        path: name to save the file ('name.pth')
        optimizer: optimizes the model
        structure: model architecture
        model: trained model
        lr: learning rate
        epochs: epochs to use
        train_set: training data for the model

    Returns:
        None
    """
    # Save the checkpoint
    model.class_to_idx = train_set.class_to_idx

    # Get the hyperparameters of the model
    model_checkpoint = {'input_size': 25088,
                        'output_size': 102,
                        'structure': structure,
                        'learning_rate': lr,
                        'classifier': model.classifier,
                        'epochs': epochs,
                        'optimizer': optimizer.state_dict(),
                        'state_dict': model.state_dict(),
                        'class_to_idx': model.class_to_idx, }

    torch.save(model_checkpoint, path)
    print("Checkpoint Saved")


def load_checkpoint(path):
    """ Function thats loads model

    Arg:
        filepath(str): path to the saved model
    Returns:
        model: the loaded model
    """

    checkpoint = torch.load(path)
    model = getattr(torchvision.models, checkpoint['structure'])(pretrained=True)

    # Freeze Parameters
    for param in model.parameters():
        param.requires_grad = False

    # Define hyperparameters
    model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    model.optimizer = checkpoint['optimizer']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model


def process_image(image_path):
    """ Function scales, crops, and normalizes a PIL image for a PyTorch model,

        Arg:
            image_path: path to the image
        Returns:
            image(an Numpy array)
    """
    # open image
    image = Image.open(image_path)

    # size of the image in pixel
    width, height = image.size
    img_size = 0, 0
    # resizing image
    if width == height:
        img_size = 256, 256
    elif width > height:
        aspect_ratio = width / height
        img_size = 256 * aspect_ratio, 256
    elif width < height:
        aspect_ratio = height / width
        img_size = 256, 256 * aspect_ratio

    image.thumbnail(img_size, Image.ANTIALIAS)

    # specify points for cropping
    left = (256 - 224) / 2
    top = (256 - 224) / 2
    right = (256 + 224) / 2
    bottom = (256 + 224) / 2

    # Crop the center of the image
    image = image.crop((left, top, right, bottom))

    # color channels converted to float between 0 and 1
    image = np.array(image)
    image = image / 255

    # normalize the images
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std

    # make color channel of the image the first dimension
    image = image.transpose((2, 0, 1))

    return image
