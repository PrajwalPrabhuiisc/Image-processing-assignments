import cv2
import numpy as np

image = cv2.imread('test5.jpeg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
bins_num = 256
hist, bin_edges = np.histogram(image, bins=bins_num)

bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2
p1 = np.cumsum(hist)
p2 = np.cumsum(hist[::-1])[::-1]
mean1 = np.cumsum(hist * bin_mids) / p1
mean2 = (np.cumsum((hist * bin_mids)[::-1]) / p2[::-1])[::-1]
inter_class_variance = p1[:-1] * p2[1:] * (mean1[:-1] - mean2[1:]) ** 2
index_of_max_val = np.argmax(inter_class_variance)
threshold = round(bin_mids[:-1][index_of_max_val])
print("Otsu's algorithm implementation thresholding result: ", threshold)
ret, thresh1 = cv2.threshold(image, 142, 255, cv2.THRESH_BINARY +
                             cv2.THRESH_OTSU)
print("Otsu's algorithm  thresholding result: ", ret)
cv2.imshow("thresholded image",thresh1)
cv2.waitKey()


