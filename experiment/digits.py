import numpy as np
import matplotlib.pyplot as plt
import keras 
from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
from keras.models import load_model
from PIL import Image
import cv2
from skimage import transform

def main():
    # get data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # data normalization / preprocessing
    x_train = x_train/255.
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test/255.
    x_test = x_test.reshape(-1, 28, 28, 1)
    y_train = to_categorical(y_train, num_classes = 10)
    model = load_model('./data/my_model.h5')

    # predicting
    results = model.predict(x_test)
    results = np.argmax(results, axis = 1)
    diff = results - y_test
    test_accuracy = 1.-(np.count_nonzero(diff)/len(diff))
    print("Test Accuracy:", test_accuracy)

def test_sudoku():
    img = cv2.imread('./experiment/digits.png')
    print(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    model = load_model('./data/my_model.h5')
    images = []
    images_idx = []
    h, w = 28, 28
    x, y = int(3+35*1), int(35+7.5)
    cnt = 0

    # cutting the image into cells
    for i in range(9):
      y = int(5.+i*35)
      if (i >= 5):
        y -= 5
      for j in range(9):
        x = int(4.+35*j)
        if (j >= 5):
          x -= 3
        crop_img = 255-gray[y:y+h, x:x+w]
        print(cnt, i, j, np.mean(crop_img[4:24,4:24]))
        if (np.mean(crop_img[4:24,4:24]) > 22.):
          images.append(crop_img)
          images_idx.append(cnt)
        cnt += 1
        

    # h, w = 28, 28
    # y = 70
    # x = 5
    crop_img = gray[y:y+h, x:x+w]
    test_image = np.array(images)/255.
    test_image = transform.resize(test_image, (len(images), 28, 28, 1))
    # test_image = np.expand_dims(test_image, axis=0)
    plt.imshow(test_image[0,:,:,0], cmap="gray")
    result = model.predict(test_image)
    # print(np.argmax(result))
    for i in range(len(images)):
      print("Predicted value:", np.argmax(result[i]))
      plt.imshow(images[i], cmap="gray")
      plt.show()


if __name__ == "__main__":
    # main()
    test_sudoku()