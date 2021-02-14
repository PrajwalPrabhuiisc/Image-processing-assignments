import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.io
from numpy.fft import fft2, ifft2

path4 = r'..\pythonProject\Blurred-LowNoise.png'
path5 = r'..\pythonProject\Blurred-MedNoise.png'
path6 = r'..\pythonProject\Blurred-HighNoise.png'


def constrained_ls_filter(img, kernel, laplacian, gamma):
    kernel /= np.sum(kernel)
    dummy = np.copy(img)
    dummy = np.fft.fftshift(np.fft.fft2(dummy))
    kernel = np.fft.fftshift(np.fft.fft2(kernel, s=img.shape))
    P = np.fft.fftshift(np.fft.fft2(laplacian, s=img.shape))
    kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + gamma * P)
    dummy = dummy * kernel
    dummy = np.abs(np.fft.ifft2(np.fft.ifftshift(dummy)))
    return dummy


P = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
gamma = 0.01
[image1, image2, image3] = cv2.imread(path4, 0), cv2.imread(path5, 0), cv2.imread(path6, 0)
mat = scipy.io.loadmat('BlurKernel.mat')
kernel = mat.get('h')
K = 0.01
[cls1, cls2, cls3] = constrained_ls_filter(image1, kernel, P, gamma), constrained_ls_filter(image2, kernel, P,
                                                                                            gamma), constrained_ls_filter(
    image3, kernel, P, gamma)
display13 = [image1, image2, image3, cls1, cls2, cls3]
# resultant images need filtering for the sake of better quality and completeness 
label13 = ['Low Noise Image', 'Med Noise Image', 'High Noise Image', 'cls-Low Noise o/p', 'cls-Med Noise o/p',
           'cls-High Noise o/p']
fig13 = plt.figure('Constrained least squares Filter for Image Deblurring', figsize=(12, 10))
for i in range(len(display13)):
    fig13.add_subplot(2, 3, i + 1)
    plt.imshow(display13[i], cmap='gray')
    plt.title(label13[i])
plt.show()
