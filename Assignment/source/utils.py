import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import math

def rAngle(img, ang): # angle in radians.
  canvas = np.zeros((int(img.shape[0] * 2), int(img.shape[1] * 2), 3), dtype=np.uint8)
  h, w, z = img.shape
  hh, ww, z = canvas.shape
  yoff = round((hh-h)/2)
  xoff = round((ww-w)/2)
  result = canvas.copy()
  result[yoff:yoff+h, xoff:xoff+w] = img
  img = result
  degrees = (ang * 180) / math.pi
  center = tuple(np.array(img.shape[1::-1]) / 2)
  rotateMatrix = cv.getRotationMatrix2D(center, degrees, 1.0)
  return cv.warpAffine(img, rotateMatrix, img.shape[1::-1], flags=cv.INTER_LINEAR)

def scaleImg(img, widthScale, heightScale):
  newW = int(img.shape[1] * widthScale)
  newH = int(img.shape[0] * heightScale)
  img = cv.resize(img, (newW, newH))
  return img

def calcHist(img, filename):
  color = ('b','g','r')
  for i, col in enumerate(color):
    histr = cv.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histr, color = col)
    plt.xlim([0, 256])
  plt.savefig(filename)
  plt.clf()

def placeOn(img, desiredWidth, desiredHeight):
  canvas = np.zeros((desiredHeight, desiredWidth), dtype=np.uint8)
  canvas[0:img.shape[0], 0:img.shape[1]] = img
  return canvas