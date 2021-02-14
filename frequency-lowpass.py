import cv2
import numpy as np


def ideallowpass(img, size, Do) -> list:
    D = np.zeros([size, size], dtype=np.uint32)
    H = np.zeros([size, size], dtype=np.uint8)
    r = img.shape[0] // 2
    c = img.shape[1] // 2
    for u in range(0, size):
        for v in range(0, size):
            D[u, v] = np.sqrt((u - r) ** 2 + (v - c) ** 2)
    for i in range(size):
        for j in range(size):
            if D[i, j] > Do:
                H[i, j] = 0
            else:
                H[i, j] = 255
    return [H, D]


image = cv2.imread('characters.tif', 0)
size = image.shape[0]
Do = 80
H, D = ideallowpass(image, size, Do)
input1 = np.fft.fftshift(np.fft.fft2(image))
out = input1*H
out = np.abs(np.fft.ifft2(np.fft.ifftshift(out)))
out = np.uint8(cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX, -1))
cv2.imshow('Filter Representation', H)
cv2.waitKey(0)
cv2.imshow('Magnitude Spectrum', cv2.normalize(np.abs(input1), None, 0, 255, cv2.NORM_MINMAX, -1))
cv2.waitKey(0)
cv2.imshow('Ideal Low Pass Filtered Output', out)
cv2.waitKey(0)
