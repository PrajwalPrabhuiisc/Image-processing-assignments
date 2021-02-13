import cv2
import numpy as np


def GaussianLowPass(img, size, Do) -> list:
    D1 = np.zeros([size, size], dtype=np.uint32)
    H1 = np.zeros([size, size], dtype=float)
    r1 = img.shape[0] // 2
    c1 = img.shape[1] // 2
    for u in range(0, size):
        for v in range(0, size):
            D1[u, v] = np.sqrt((u - r1) ** 2 + (v - c1) ** 2)
    for i in range(size):
        for j in range(size):
            H1[i, j] = np.exp(-D1[i, j] ** 2 / (2 * Do) ** 2)

    return [H1, D1]


Do1 = 100
img1 = cv2.imread('noisy.tif', 0)
size1 = img1.shape[0]
H1, D1 = GaussianLowPass(img1, size1, Do1)
input2 = np.fft.fftshift(np.fft.fft2(img1))
out1 = H1 * input2
out1 = np.abs(np.fft.ifft2(np.fft.ifftshift(out1)))
out1 = np.uint8(cv2.normalize(out1, None, 0, 255, cv2.NORM_MINMAX, -1))
cv2.imshow('Filter Representation', H1)
cv2.waitKey(0)
cv2.imshow('Magnitude Spectrum', cv2.normalize(np.abs(input2), None, 0, 255, cv2.NORM_MINMAX, -1))
cv2.waitKey(0)
cv2.imshow('Gaussian Low Pass Filtered Output', out1)
cv2.waitKey(0)
cv2.imwrite('Gauss-lowpass.png', out1)
