import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt


def ReadImage(img_path):
    img_in = cv2.imread(img_path)
    img = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    return img.astype(float) / 255.0, img_gray


def EdgeDetection(img_gray):
    return cv2.Canny(img_gray, 100, 200)


if __name__ == '__main__':
    # TODO: use argument

    img, img_gray = ReadImage(sys.argv[1])
    img_edge = EdgeDetection(img_gray)

    # TODO: split grid, digit recognition, solve sudoku

    #############################

    fig = plt.figure(dpi=150)
    a = fig.add_subplot(1, 2, 1)
    imgplot = plt.imshow(img)

    a = fig.add_subplot(1, 2, 2)
    imgplot = plt.imshow(img_edge, cmap='gray')

    plt.show()
