import cv2 as cv
import numpy as np
import utils
import math
from matplotlib import pyplot as plt
# 860 x 480 480 x 274
hog = cv.HOGDescriptor('hog.xml')
winStride = (8,8)
padding = (8,8)
locations = ((0,0),)
card = cv.imread('./images/diamond2.png', cv.IMREAD_GRAYSCALE)
card90 = cv.rotate(card, cv.ROTATE_90_CLOCKWISE)
cardn90 = cv.rotate(card, cv.ROTATE_90_COUNTERCLOCKWISE)
card180 = cv.rotate(card, cv.ROTATE_180)
dugong = cv.imread('./images/Dugong.jpg', cv.IMREAD_GRAYSCALE)
card = card[5:69, 0:64] # crop to 64x64 number 2 feature.
dugong = dugong[209:209+64, 390:390+64] # crop to 64x64 dugong feature. 
dugong90 = cv.rotate(dugong, cv.ROTATE_90_CLOCKWISE)
dugongn90 = cv.rotate(dugong, cv.ROTATE_90_COUNTERCLOCKWISE)
dugong180 = cv.rotate(dugong, cv.ROTATE_180)
cv.imshow('card', card)
cv.imshow('dugong', dugong)
cardHists = [
  (hog.compute(card,winStride,padding,locations), './out_files/task_2/rotation/noRotation-card.png'), (hog.compute(card90,winStride,padding,locations), './out_files/task_2/rotation/90deg-card.png'), 
  (hog.compute(cardn90,winStride,padding,locations), './out_files/task_2/rotation/n90deg-card.png'), (hog.compute(card180,winStride,padding,locations), './out_files/task_2/rotation/180deg-card.png')
]
dugongHists = [
  hog.compute(dugong,winStride,padding,locations), hog.compute(dugong90,winStride,padding,locations),
  hog.compute(dugongn90,winStride,padding,locations), hog.compute(dugong180,winStride,padding,locations)
]
for hist in cardHists:
  plt.plot(hist[0])
  plt.savefig(hist[1])