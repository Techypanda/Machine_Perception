import cv2 as cv
import numpy as np
import utils
import math
from matplotlib import pyplot as plt
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
cardHog = [
  (hog.compute(card,winStride,padding,locations), 'No Rotation'),
  (hog.compute(card90,winStride,padding,locations), '90 Degree Rotation'), 
  (hog.compute(cardn90,winStride,padding,locations), '-90 Degree Rotation'),
  (hog.compute(card180,winStride,padding,locations), '180 Degree Rotation')
]
dugongHog = [
  (hog.compute(dugong,winStride,padding,locations), 'No Rotation'),
  (hog.compute(dugong90,winStride,padding,locations), '90 Degree Rotation'),
  (hog.compute(dugongn90,winStride,padding,locations), '-90 Degree Rotation'),
  (hog.compute(dugong180,winStride,padding,locations), '180 Degree Rotation')
]
with open("./out_files/task_2/rotation/cardHogResults.txt", "w") as f:
  for h in cardHog:
    f.write("{0} = {1}\n".format(h[1], 
      utils.hogVariation(cardHog[0][0][0], h[0][0][0])))
with open("./out_files/task_2/rotation/dugongHogResults.txt", "w") as f:
  for h in dugongHog:
    f.write("{0} = {1}\n".format(h[1], 
      utils.hogVariation(dugongHog[0][0][0], h[0][0][0])))

scaledDugongs = [dugong, utils.placeOn(utils.scaleImg(dugong, 0.50, 0.50), 64, 64),
                 utils.placeOn(utils.scaleImg(dugong, 0.25, 0.25), 64, 64), 
                 utils.placeOn(utils.scaleImg(dugong, 0.50, 1), 64, 64), 
                 utils.placeOn(utils.scaleImg(dugong, 1, 0.50), 64, 64)]
scaledCards = [card, utils.placeOn(utils.scaleImg(card, 0.50, 0.50), 64, 64),
               utils.placeOn(utils.scaleImg(card, 0.25, 0.25), 64, 64),
               utils.placeOn(utils.scaleImg(card, 0.50, 1), 64, 64),
               utils.placeOn(utils.scaleImg(card, 1, 0.50), 64, 64)]
cardHog = [
  (hog.compute(card,winStride,padding,locations), 'No Scale'),
  (hog.compute(scaledCards[1],winStride,padding,locations), '50%_y 50%_x'), 
  (hog.compute(scaledCards[2],winStride,padding,locations), '25%_y 25%_x'),
  (hog.compute(scaledCards[3],winStride,padding,locations), '100%_x 50%_y'),
  (hog.compute(scaledCards[4],winStride,padding,locations), '100%_y 50%_x')
]
dugongHog = [
  (hog.compute(dugong,winStride,padding,locations), 'No Scale'),
  (hog.compute(scaledDugongs[1],winStride,padding,locations), '50%_x 50%_y'), 
  (hog.compute(scaledDugongs[2],winStride,padding,locations), '25%_x 25%_y'),
  (hog.compute(scaledDugongs[3],winStride,padding,locations), '50%_x 100%_y'),
  (hog.compute(scaledDugongs[4],winStride,padding,locations), '100%_x 50%_y')
]
with open("./out_files/task_2/scaled/cardHogResults.txt", "w") as f:
  for h in cardHog:
    f.write("{0} = {1}\n"
      .format(h[1], utils.hogVariation(cardHog[0][0][0], h[0][0][0])))
with open("./out_files/task_2/scaled/dugongHogResults.txt", "w") as f:
  for h in dugongHog:
    f.write("{0} = {1}\n"
      .format(h[1], utils.hogVariation(dugongHog[0][0][0], h[0][0][0])))

# Sift
SIFT = cv.xfeatures2d.SIFT_create()
cardHists = [
  (SIFT.detectAndCompute(card, None), 
  './out_files/task_2/rotation/noRotation-card-SIFT.png'), 
  (SIFT.detectAndCompute(card90, None), 
  './out_files/task_2/rotation/90deg-card-SIFT.png'),
  (SIFT.detectAndCompute(cardn90, None), 
  './out_files/task_2/rotation/n90deg-card-SIFT.png'), 
  (SIFT.detectAndCompute(card180, None), 
  './out_files/task_2/rotation/180deg-card-SIFT.png')
]
dugongHists = [
  (SIFT.detectAndCompute(dugong, None), 
  './out_files/task_2/rotation/noRotation-dugong-SIFT.png'), 
  (SIFT.detectAndCompute(dugong90, None), 
  './out_files/task_2/rotation/90deg-dugong-SIFT.png'),
  (SIFT.detectAndCompute(dugongn90, None), 
  './out_files/task_2/rotation/n90deg-dugong-SIFT.png'), 
  (SIFT.detectAndCompute(dugong180, None), 
  './out_files/task_2/rotation/180deg-dugong-SIFT.png')
]
for hist in cardHists + dugongHists:
  plt.plot(hist[0][1])
  plt.savefig(hist[1])
  plt.clf()
cardHists = [
  (SIFT.detectAndCompute(scaledCards[0], None), 
  './out_files/task_2/scaled/noScale-card-sift.png'), 
  (SIFT.detectAndCompute(scaledCards[1], None), 
  './out_files/task_2/scaled/50w50h-card-sift.png'), 
  (SIFT.detectAndCompute(scaledCards[2], None), 
  './out_files/task_2/scaled/25w25h-card-sift.png'), 
  (SIFT.detectAndCompute(scaledCards[3], None), 
  './out_files/task_2/scaled/50w1h-card-sift.png'),
  (SIFT.detectAndCompute(scaledCards[4], None), 
  './out_files/task_2/scaled/1w50h-card-sift.png')
]
dugongHists = [
  (SIFT.detectAndCompute(scaledDugongs[0], None), 
  './out_files/task_2/scaled/noScale-dugong-sift.png'), 
  (SIFT.detectAndCompute(scaledDugongs[1], None), 
  './out_files/task_2/scaled/50w50h-dugong-sift.png'), 
  (SIFT.detectAndCompute(scaledDugongs[3], None), 
  './out_files/task_2/scaled/50w1h-dugong-sift.png'),
  (SIFT.detectAndCompute(scaledDugongs[4], None), 
  './out_files/task_2/scaled/1w50h-dugong-sift.png')
]
for hist in dugongHists:
  plt.plot(hist[0][1])
  plt.savefig(hist[1])
  plt.clf()