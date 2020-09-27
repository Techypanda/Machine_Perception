import cv2 as cv
import numpy as np
import utils
import math
from matplotlib import pyplot as plt
utils.setup()

hog = cv.HOGDescriptor('hog.xml')
winStride = (8,8)
padding = (8,8)
locations = ((0,0),)
card = cv.imread('./images/diamond2.png', cv.IMREAD_GRAYSCALE)
card = card[5:69, 0:64] # crop to 64x64 number 2 feature.
card90 = cv.rotate(card, cv.ROTATE_90_CLOCKWISE)
cardn90 = cv.rotate(card, cv.ROTATE_90_COUNTERCLOCKWISE)
card180 = cv.rotate(card, cv.ROTATE_180)
dugong = cv.imread('./images/Dugong.jpg', cv.IMREAD_GRAYSCALE)
dugong = dugong[209:209+64, 390:390+64] # crop to 64x64 dugong feature. 
dugong90 = cv.rotate(dugong, cv.ROTATE_90_CLOCKWISE)
dugongn90 = cv.rotate(dugong, cv.ROTATE_90_COUNTERCLOCKWISE)
dugong180 = cv.rotate(dugong, cv.ROTATE_180)
cardHog = [
  (hog.compute(card,winStride,padding,locations), 'No Rotation'), (hog.compute(card90,winStride,padding,locations), '90 Degree Rotation'), 
  (hog.compute(cardn90,winStride,padding,locations), '-90 Degree Rotation'), (hog.compute(card180,winStride,padding,locations), '180 Degree Rotation')
]
dugongHog = [
  (hog.compute(dugong,winStride,padding,locations), 'No Rotation'), (hog.compute(dugong90,winStride,padding,locations), '90 Degree Rotation'),
  (hog.compute(dugongn90,winStride,padding,locations), '-90 Degree Rotation'), (hog.compute(dugong180,winStride,padding,locations), '180 Degree Rotation')
]
H0 = cardHog[0][0]
with open("./out_files/task_2/rotation/cardHogResults.txt", "w") as f:
  for h in cardHog:
    f.write("{0} = {1}\n".format(h[1], utils.hogVariation(H0, h[0])))
H0 = dugongHog[0][0]
with open("./out_files/task_2/rotation/dugongHogResults.txt", "w") as f:
  for h in dugongHog:
    f.write("{0} = {1}\n".format(h[1], utils.hogVariation(H0, h[0])))

scaledDugongs = [dugong, utils.placeOn(utils.scaleImg(dugong, 0.50, 0.50), 64, 64), utils.placeOn(utils.scaleImg(dugong, 0.25, 0.25), 64, 64), utils.placeOn(utils.scaleImg(dugong, 0.50, 1), 64, 64), utils.placeOn(utils.scaleImg(dugong, 1, 0.50), 64, 64)]
scaledCards = [card, utils.placeOn(utils.scaleImg(card, 0.50, 0.50), 64, 64), utils.placeOn(utils.scaleImg(card, 0.25, 0.25), 64, 64), utils.placeOn(utils.scaleImg(card, 0.50, 1), 64, 64), utils.placeOn(utils.scaleImg(card, 1, 0.50), 64, 64)]
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
H0 = cardHog[0][0]
with open("./out_files/task_2/scaled/cardHogResults.txt", "w") as f:
  for h in cardHog:
    f.write("{0} = {1}\n".format(h[1], utils.hogVariation(H0, h[0])))
H0 = dugongHog[0][0]
with open("./out_files/task_2/scaled/dugongHogResults.txt", "w") as f:
  for h in dugongHog:
    f.write("{0} = {1}\n".format(h[1], utils.hogVariation(H0, h[0])))

# SIFT
cards = [card, card90, cardn90, card180]
dugongs = [dugong, dugong90, dugongn90, dugong180]
filenames = ['card-sift', 'card90-sift', 'cardn90-sift', 'card180-sift']; i=0
for c in cards:
  SIFT = cv.xfeatures2d_SIFT.create()
  kp, des = SIFT.detectAndCompute(c, None)
  out = cv.drawKeypoints(c, kp, c)
  cv.imwrite('./out_files/task_2/rotation/'+filenames[i]+'.png',out)
  i += 1
filenames = ['dugong-sift', 'dugong90-sift', 'dugongn90-sift', 'dugong180-sift']; i=0
for d in dugongs:
  SIFT = cv.xfeatures2d_SIFT.create()
  kp, des = SIFT.detectAndCompute(d, None)
  out = cv.drawKeypoints(d, kp, d)
  cv.imwrite('./out_files/task_2/rotation/'+filenames[i]+'.png',out)
  i += 1
filenames = ['cardScaled-sift', 'card50pc-sift', 'card25pc-sift', 'card50w-sift', 'card50h-sift']; i=0
for c in scaledCards:
  SIFT = cv.xfeatures2d_SIFT.create()
  kp, des = SIFT.detectAndCompute(c, None)
  out = cv.drawKeypoints(c, kp, c)
  cv.imwrite('./out_files/task_2/scaled/'+filenames[i]+'.png',out)
  i += 1
filenames = ['dugongScaled-sift', 'dugong50pc-sift', 'dugong25pc-sift', 'dugong50w-sift', 'dugong50w-sift']; i=0
for d in scaledDugongs:
  SIFT = cv.xfeatures2d_SIFT.create()
  kp, des = SIFT.detectAndCompute(d, None)
  out = cv.drawKeypoints(d, kp, d)
  cv.imwrite('./out_files/task_2/scaled/'+filenames[i]+'.png',out)
  i += 1