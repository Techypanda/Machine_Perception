import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img = cv.imread('./in/digits.png')

zero = img[0:20, 0:img.shape[0]]
one = img[20*5:20*6, 0:img.shape[0]]
two = img[20*10:20*11, 0:img.shape[0]]
three = img[20*15:20*16, 0:img.shape[0]]
four = img[20*20:20*21, 0:img.shape[0]]
five = img[20*25:20*26, 0:img.shape[0]]
six = img[20*30:20*31, 0:img.shape[0]]
seven = img[20*35:20*36, 0:img.shape[0]]
eight = img[20*40:20*41, 0:img.shape[0]]
nine = img[20*45:20*46, 0:img.shape[0]]
rows = [zero, one, two, three, four, five, six, seven, eight, nine]
for row in rows:
  compressed = row[0:img.shape[1], 0:20]
  for x in range(20, img.shape[0], 20):
    n = row[0:img.shape[1], x:x+20]
    compressed = np.hstack((compressed, n))
  compressed = np.float32(compressed.reshape((1, 60000)))
  plt.hist(compressed, 256, [0,256])
  plt.show()
  plt.clr()

#cv.waitKey()