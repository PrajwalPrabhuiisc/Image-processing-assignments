import numpy as np
from scipy import ndimage
import skimage.measure


def convolve2d(image, kernel):
    kernel = np.flipud(np.fliplr(kernel))
    output = np.zeros_like(image)
    image_padded = np.zeros((image.shape[0] + 2, image.shape[1] + 2))
    image_padded[1:-1, 1:-1] = image
    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            # element-wise multiplication of the kernel and the image
            output[y, x] = (kernel * image_padded[y: y + 3, x: x + 3]).sum()

    return output


a = np.array([
    [111,222,163,230,224,217],
    [225,32,110,215,187,88],
    [41,92,184,97,161,16],
    [194,233,127,202,29,194],
    [166,141,164,172,237,99],
    [72,66,204,8,225,139]
])
b = np.array([[0, 1, 0],
              [1, 0, 1],
              [0, 1, 0]
              ])
conv = convolve2d(a, b)
conv1 = ndimage.convolve(a, b, mode='constant', cval=0.0)
maxpool = skimage.measure.block_reduce(a, (2, 2), np.max)
avgpool = skimage.measure.block_reduce(a, (2, 2), np.average)
print("maxpool=", maxpool)
print("convolution=", conv)
print("avgpool=", avgpool)
