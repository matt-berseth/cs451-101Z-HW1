import logging
import os
import warnings

from sklearn.datasets import fetch_openml
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def main():
    logging.info("Checking if mnist data exists ...")
    if not os.path.exists("./.data"):
        logging.info("mnist data does not exist, downloading and saving locally ...")
        mnist = fetch_openml("mnist_784")

        # for now, only interested in 10,000 images
        n = 10000
        os.makedirs("./.data", exist_ok=True)
        np.savetxt("./.data/y.txt", mnist.target[:n], fmt="%s")
        np.savetxt("./.data/X.txt", mnist.data[:n], fmt="%1.2f")

    logging.info("loading mnist data ...")
    X = np.loadtxt("./.data/X.txt")
    y = np.loadtxt("./.data/y.txt")
    logging.info(f"X.shape: {X.shape}, y.shape: {y.shape}")



    # TODO(Task #1): reshape each example to 28x28 matrix. Plot a digit from
    # each class using matplotlib. Save an example of each digit as a .png file. After
    # you complete this, you should have 10 .png files. 0.png, 1.png, 2.png ... 9.png.
    # In your report, include an image for each digit.
    
    # HINT: the following code snippet pulls out the first example
    # from the training data and saves it as a png file.
    x0 = X[0] # pull out a single 1x784 array
    x0 = x0.reshape(28, 28) # reshape into a 28x28 matrix
    y0 = y[0] # get the matching label for the first image
    logging.info(f"x.shape: {x0.shape}, label: {y0}")
    plt.imshow(x0, cmap=matplotlib.cm.binary, interpolation="nearest")
    plt.axis("off")
    plt.savefig(f"./{int(y0)}.png")



    # TODO(Task #2): provide the number of examples of each class.
    # i.e.
    # digit 0: ?
    # digit 1: ?
    # ...
    # digit 9: ?

    # HINT: iterate over the values in the 'y' array and sum up the number for each
    # of the classes.



    # TODO(Task #3): for each digit, what percent of the pixels have a value greater 
    # than zero?
    # i.e.
    # digit 0: ? pixel values out of ? have a value greater than zero
    # digit 1: ? pixel values out of ? have a value greater than zero
    # ...
    # digit 9: ? pixel values out of ? have a value greater than zero

    # HINT: iterate over the values in the 'y' array and sum up the number for each
    # of the classes.



    # TODO(Task #4): train a DecisionTreeClassifier model using 'X' and 'y'
    # Report the classification accuracy of the model using the training data ('X' and 
    # 'y')

    # HINT: Review the documentation here for available options: 
    # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html



    # TODO(Task #5): train a RandomForestClassifier model using 'X' and 'y'
    # Report the classification accuracy of the model using the training data ('X' and 
    # 'y')

    # HINT: Review the documentation here for available options: 
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
