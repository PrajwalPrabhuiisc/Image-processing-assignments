import numpy as np
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt
from pathlib import Path


def window(kernel):
    (lpfw, lpfh) = (kernel, kernel)
    lowPassFilter = np.ones((lpfw, lpfh)) * 1 / (lpfw * lpfh)
    return lowPassFilter


def mse(imageA, imageB):
    err = np.mean(np.square(imageA - imageB))
    return err


def highboost(img1: Path, img2: Path, kernelsize, boostingvalue) -> list:
    img = cv2.imread(img1, 0)
    img1 = cv2.imread(img2, 0)
    output = []
    errors = []
    filter = window(kernelsize)
    img_rst = ndimage.convolve(img, filter, mode='constant', cval=0.0)
    img_blur = ndimage.convolve(img_rst, filter, mode='reflect')
    blur_mask = img_rst - img_blur
    for i in range(0, len(boostingvalue)):
        high_boost_image = img_rst + boostingvalue[i] * blur_mask
        output.append(high_boost_image)
        error = mse(output[i], img1)
        errors.append(error)
        fig = plt.figure('high-boost image with mask = %d,' % kernelsize)
        plt.suptitle("MSE: %.1f, " % error)
        ax = fig.add_subplot(1, 2, 1)
        plt.imshow(img_rst, cmap=plt.cm.gray)
        plt.title('Filtered Image,window size=%d,' % kernelsize)
        plt.axis("off")
        ax = fig.add_subplot(1, 2, 2)
        plt.imshow(high_boost_image, cmap=plt.cm.gray)
        plt.title('High boosted image with k = %d,' % boostingvalue[i])
        plt.axis("off")
        plt.show()
    return [img_rst, output, errors]


path = r'..\pythonProject\noisy.tif'
path1 = r'..\pythonProject\characters.tif'

kernel_size = 10
boostingvalue = list(range(-30, 40, 10))
img_rst, high_boost_image, error = highboost(path, path1, kernel_size, boostingvalue)
plt.plot(boostingvalue, error, color='green', linestyle='dashed', linewidth=3,
         marker='o', markerfacecolor='blue', markersize=12)
plt.xlabel('Scaling Constant-K')
plt.ylabel('Mean squared error')
plt.title('Plot of Scaling Constant k v/s Mean squared error')
plt.show()
