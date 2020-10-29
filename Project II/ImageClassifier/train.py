import torch
import time
import argparse

from datapro import load_data, save_checkpoint
from modfun import network, validation

pre_models = {"vgg16": 25088, "densenet121": 1024}
parser = argparse.ArgumentParser(description='Parser For Training')

parser.add_argument('data_dir', action="store", default="./flowers/")
parser.add_argument('--save_dir', action="store", default="./checkpoint.pth")
parser.add_argument('--pre_models', action="store", default="vgg16")
parser.add_argument('--learning_rate', action="store", type=float, default=0.001)
parser.add_argument('--hidden_units', action="store", dest="hidden_units", type=int, default=[2048, 256])
parser.add_argument('--epochs', action="store", default=3, type=int)
parser.add_argument('--gpu', action="store", default="gpu")

args = parser.parse_args()
dir_path = args.data_dir
path = args.save_dir
lr = args.learning_rate
structure = args.pre_models
hidden_units = args.hidden_units
gpu = args.gpu
epochs = args.epochs


def main():
    """ Function that runs the main training process
    Arg:
        None
    Returns:
        None
    """
    trainloader, testloader, validloader, train_set = load_data(dir_path)
    model, criterion, optimizer = network(structure, hidden_units, lr, gpu)

    # Train the classifier layers using backprop
    steps = 0
    print_every = 20
    running_loss = 0

    # Submit model to cuda for faster processing
    if gpu == 'gpu':
        model.to('cuda')
    else:
        model.to('cpu')

    # Time processing
    start = time.time()
    print('Training started:')

    for e in range(epochs):

        for images, labels in trainloader:
            steps += 1

            # Submit images and labels tensor to cuda
            if gpu == 'gpu':
                images, labels = images.to('cuda'), labels.to('cuda')
            else:
                images, labels = images.to('cpu'), labels.to('cpu')

            optimizer.zero_grad()

            # Perform forward pass through the network
            output_probs = model.forward(images)
            loss = criterion(output_probs, labels)

            # Perform Backwardprop
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()

                with torch.no_grad():
                    # valid_loss, accuracy = validation(model, validloader, criterion)
                    valid_loss, accuracy = validation(model, validloader, criterion)

                print("Epoch:{}/{}.... ".format(e + 1, epochs),
                      "Loss: {:.3f}.... ".format(running_loss / print_every),
                      "Validation Loss: {:.3f}.... ".format(valid_loss / len(validloader)),
                      "Accuracy: {:.3f}".format(accuracy))

                running_loss = 0

                model.train()

    time_elapsed = time.time() - start
    print("\nTraining Time: {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

    save_checkpoint(path, optimizer, structure, model, lr, epochs, train_set)

if __name__ == '__main__':
    main()
