import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.io
from numpy.fft import fft2, ifft2
path4 = r'..\pythonProject\Blurred-LowNoise.png'
path5 = r'..\pythonProject\Blurred-MedNoise.png'
path6 = r'..\pythonProject\Blurred-HighNoise.png'


def inverse_filter(img, kernel,k):
    kernel /= np.sum(kernel)
    dummy = np.copy(img)
    dummy = np.fft.fftshift(np.fft.fft2(dummy))
    kernel = np.fft.fftshift(np.fft.fft2(kernel, s=img.shape))
    kernel = np.conj(kernel) / (np.abs(kernel) ** 2+k)
    dummy = dummy * kernel
    dummy = np.abs(np.fft.ifft2(np.fft.ifftshift(dummy)))
    return dummy

[image1, image2, image3] = cv2.imread(path4, 0), cv2.imread(path5, 0), cv2.imread(path6, 0)
mat = scipy.io.loadmat('BlurKernel.mat')
kernel = mat.get('h')
K = 0.01
[inv1, inv2, inv3] = inverse_filter(image1, kernel, K), inverse_filter(image2, kernel, K), inverse_filter(image3,kernel, K)
# Image was deblurred from the input. For the sake of completeness need a smoothing filter to ensure the image obtained is more clean
display11 = [image1, image2, image3, inv1, inv2, inv3]
label11 = ['Low Noise Image', 'Med Noise Image', 'High Noise Image', 'inv-Low Noise o/p', 'inv-Med Noise o/p',
           'inv-High Noise o/p']
fig11 = plt.figure('Inverse Filter for Image Deblurring', figsize=(12, 10))
for i in range(len(display11)):
    fig11.add_subplot(2, 3, i + 1)
    plt.imshow(display11[i], cmap='gray')
    plt.title(label11[i])
plt.show()
