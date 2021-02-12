import cv2
import matplotlib.pyplot as plt

image = cv2.imread('gray-1.png')
image = cv2.resize(image, (300, 300))
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
m, n = image.shape
count = 0
frequency = [0] * 256
for i in range(1, 256):
    for j in range(1, m):
        for k in range(1, n):
            if image[j][k] == i - 1:
                count = count + 1
    frequency[i] = count
    count = 0


def createList(r1, r2):
    return [item for item in range(r1, r2 + 1)]


r1, r2 = 0, 255

x = createList(r1, r2)
plt.stem(x, frequency, use_line_collection=True)
plt.title('Histogram for the given image')
plt.show()
