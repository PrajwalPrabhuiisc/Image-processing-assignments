import cv2
import matplotlib.pyplot as plt

class Histogram:
    def __init__(self, image, resize, r1, r2):
        self.image = image
        self.resize = resize
        self.r1, self.r2 = r1, r2

    def plot(self):
        image1 = cv2.resize(self.image, (self.resize, self.resize))
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        m, n = image1.shape
        count = 0
        frequency = [0] * 256
        for i in range(1, 256):
            for j in range(1, m):
                for k in range(1, n):
                    if image1[j][k] == i - 1:
                        count = count + 1
            frequency[i] = count
            count = 0
        return frequency

    def createList(self):
        return [item for item in range(self.r1,self.r2 + 1)]

if __name__ == '__main__':
    image = cv2.imread('test_image.jpg')
    resize = 300
    r1 = 0
    r2 = 255
    hist = Histogram(image, resize, r1, r2)
    f = hist.plot()
    x = hist.createList()
    plt.stem(x, f, use_line_collection=True)
    plt.title('Histogram for the given image')
    plt.show()
    
