import numpy as np
import argparse
from datapro import load_checkpoint, load_image_names
from modfun import predict

parser = argparse.ArgumentParser(description='Prediction parser')

parser.add_argument('image_path', default='./flowers/test/1/image_06752.jpg', action="store", type=str)
parser.add_argument('--dir', action="store", default="./flowers/")
parser.add_argument('checkpoint', default='./checkpoint.pth', nargs='?', action="store", type=str)
parser.add_argument('--top_k', type=int, default=5, action="store", help="Number of classes to predict")
parser.add_argument('--category_names', type=str, action="store", default='cat_to_name.json')
parser.add_argument('--gpu', default="gpu", action="store", help='Use either gpu or cpu')

args = parser.parse_args()
image_path = args.image_path
topk = args.top_k
filename = args.category_names
gpu = args.gpu
path = args.checkpoint


def main():
    model = load_checkpoint(path)
    cat_to_name = load_image_names(filename)

    probs = predict(image_path, model, topk)

    # Get the probabilities and names of predicted flowers
    prediction = np.array(probs[0])
    classes = np.array(probs[1])
    names = [cat_to_name[str(index)] for index in classes]

    # Print likely image classes and associated probabilities
    print("Likely image classes: {}".format(classes))
    print("Associated probabilities: {}".format(prediction))

    # Print the top K classes along with associated probabilities
    i = 0
    while i < topk:
        print("{} with a probability of {}".format(names[i], prediction[i]))
        i += 1

    print("Prediction Complete")


if __name__ == '__main__':
    main()
