import cv2
import numpy as np


def HomomorphicFilter (img,size,Do,gammah,gammal)->list:
    D2 = np.zeros([size, size], dtype=np.uint32)
    H2 = np.zeros([size, size], dtype=float)
    r2= img.shape[0] // 2
    c2 = img.shape[1] // 2
    for u in range(0, size):
        for v in range(0, size):
            D2[u, v] = np.sqrt((u - r2)**2 + (v - c2)**2)
    for i in range(size):
        for j in range(size):
            H2[i, j] = (gammah-gammal)*(1-np.exp(-D2[i,j]**2/(2*Do)**2))+ gammal
    return [H2, D2]


Do2 = 100
img2 = cv2.imread('PET_image.tif', 0)
img2 = cv2.resize(img2, (1610, 1610))
size2 = img2.shape[0]
GammaH =3
GammaL = 0.4
H2, D2 = HomomorphicFilter(img2, size2, Do2 ,GammaH , GammaL)
input3 = np.fft.fftshift(np.fft.fft2(img2))
out2 = H2 * input3
out2 = np.abs(np.fft.ifft2(np.fft.ifftshift(out2)))
out2 = np.uint8(cv2.normalize(out2, None, 0, 255, cv2.NORM_MINMAX, -1))
H2= cv2.resize(H2, (500,500))
cv2.imshow('Filter Representation', H2)
cv2.waitKey(0)
cv2.imshow('Magnitude Spectrum', cv2.normalize(np.abs(input3), None, 0, 255, cv2.NORM_MINMAX, -1))
cv2.waitKey(0)
out2 = cv2.resize(out2, (746, 746))
cv2.imshow('Homomorphic Filtered Output', out2)
cv2.waitKey(0)
cv2.imwrite('Homomophic results.png', out2)
