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
cardHog = [
  (hog.compute(card,winStride,padding,locations), './out_files/task_2/rotation/noRotation-card-hog.png'), (hog.compute(card90,winStride,padding,locations), './out_files/task_2/rotation/90deg-card-hog.png'), 
  (hog.compute(cardn90,winStride,padding,locations), './out_files/task_2/rotation/n90deg-card-hog.png'), (hog.compute(card180,winStride,padding,locations), './out_files/task_2/rotation/180deg-card-hog.png')
]
dugongHog = [
  (hog.compute(dugong,winStride,padding,locations), './out_files/task_2/rotation/noRotation-dugong-hog.png'), (hog.compute(dugong90,winStride,padding,locations), './out_files/task_2/rotation/90deg-dugong-hog.png'),
  (hog.compute(dugongn90,winStride,padding,locations), './out_files/task_2/rotation/n90-dugong-hog.png'), (hog.compute(dugong180,winStride,padding,locations), './out_files/task_2/rotation/180deg-dugong-hog.png')
]
for results in cardHog + dugongHog:
  print(cv.HOGDescriptor.getDescriptorSize(results)) # length wtf.
  
scaledDugongs = [dugong, utils.placeOn(utils.scaleImg(dugong, 0.50, 0.50), 64, 64), utils.placeOn(utils.scaleImg(dugong, 0.25, 0.25), 64, 64), utils.placeOn(utils.scaleImg(dugong, 0.50, 1), 64, 64), utils.placeOn(utils.scaleImg(dugong, 1, 0.50), 64, 64)]
scaledCards = [card, utils.placeOn(utils.scaleImg(card, 0.50, 0.50), 64, 64), utils.placeOn(utils.scaleImg(card, 0.25, 0.25), 64, 64), utils.placeOn(utils.scaleImg(card, 0.50, 1), 64, 64), utils.placeOn(utils.scaleImg(card, 1, 0.50), 64, 64)]
cardHog = [
  (hog.compute(card,winStride,padding,locations), './out_files/task_2/scaled/noScale-card-hog.png'), (hog.compute(scaledCards[1],winStride,padding,locations), './out_files/task_2/scaled/50w50h-card-hog.png'), 
  (hog.compute(scaledCards[2],winStride,padding,locations), './out_files/task_2/scaled/25w25h-card-hog.png'), (hog.compute(scaledCards[3],winStride,padding,locations), './out_files/task_2/scaled/50w1h-card-hog.png'),
  (hog.compute(scaledCards[4],winStride,padding,locations), './out_files/task_2/scaled/1w50h-card-hog.png')
]
dugongHog = [
  (hog.compute(dugong,winStride,padding,locations), './out_files/task_2/scaled/noScale-dugong-hog.png'), (hog.compute(scaledDugongs[1],winStride,padding,locations), './out_files/task_2/scaled/50w50h-dugong-hog.png'), 
  (hog.compute(scaledDugongs[2],winStride,padding,locations), './out_files/task_2/scaled/25w25h-dugong-hog.png'), (hog.compute(scaledDugongs[3],winStride,padding,locations), './out_files/task_2/scaled/50w1h-dugong-hog.png'),
  (hog.compute(scaledDugongs[4],winStride,padding,locations), './out_files/task_2/scaled/1w50h-dugong-hog.png')
]
for results in cardHog + dugongHog:
  plt.plot(hist[0])
  plt.savefig(hist[1])
  plt.clf()

# Sift
SIFT = cv.xfeatures2d.SIFT_create()
cardHists = [
  (SIFT.detectAndCompute(card, None), './out_files/task_2/rotation/noRotation-card-SIFT.png'), (SIFT.detectAndCompute(card90, None), './out_files/task_2/rotation/90deg-card-SIFT.png'),
  (SIFT.detectAndCompute(cardn90, None), './out_files/task_2/rotation/n90deg-card-SIFT.png'), (SIFT.detectAndCompute(card180, None), './out_files/task_2/rotation/180deg-card-SIFT.png')
]
dugongHists = [
  (SIFT.detectAndCompute(dugong, None), './out_files/task_2/rotation/noRotation-dugong-SIFT.png'), (SIFT.detectAndCompute(dugong90, None), './out_files/task_2/rotation/90deg-dugong-SIFT.png'),
  (SIFT.detectAndCompute(dugongn90, None), './out_files/task_2/rotation/n90deg-dugong-SIFT.png'), (SIFT.detectAndCompute(dugong180, None), './out_files/task_2/rotation/180deg-dugong-SIFT.png')
]
for hist in cardHists + dugongHists:
  plt.plot(hist[0][1])
  plt.savefig(hist[1])
  plt.clf()
cardHists = [
  (SIFT.detectAndCompute(scaledCards[0], None), './out_files/task_2/scaled/noScale-card-sift.png'), (SIFT.detectAndCompute(scaledCards[1], None), './out_files/task_2/scaled/50w50h-card-sift.png'), 
  (SIFT.detectAndCompute(scaledCards[2], None), './out_files/task_2/scaled/25w25h-card-sift.png'), (SIFT.detectAndCompute(scaledCards[3], None), './out_files/task_2/scaled/50w1h-card-sift.png'),
  (SIFT.detectAndCompute(scaledCards[4], None), './out_files/task_2/scaled/1w50h-card-sift.png')
]
dugongHists = [
  (SIFT.detectAndCompute(scaledDugongs[0], None), './out_files/task_2/scaled/noScale-dugong-sift.png'), (SIFT.detectAndCompute(scaledDugongs[1], None), './out_files/task_2/scaled/50w50h-dugong-sift.png'), 
  (SIFT.detectAndCompute(scaledDugongs[3], None), './out_files/task_2/scaled/50w1h-dugong-sift.png'),
  (SIFT.detectAndCompute(scaledDugongs[4], None), './out_files/task_2/scaled/1w50h-dugong-sift.png')
] # 25w 25h reutrned none
for hist in dugongHists:
  plt.plot(hist[0][1])
  plt.savefig(hist[1])
  plt.clf()