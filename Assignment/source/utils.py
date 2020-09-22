import cv2 as cv
import numpy as np
import math

def rAngle(img, ang): # angle in radians.
  degrees = (ang * 180) / math.pi
  center = tuple(np.array(img.shape[1::-1]) / 2)
  rotateMatrix = cv.getRotationMatrix2D(center, degrees, 1.0)
  return cv.warpAffine(img, rotateMatrix, img.shape[1::-1], flags=cv.INTER_LINEAR)

def scaleImg(img, widthScale, heightScale):
  newW = int(img.shape[1] * widthScale)
  newH = int(img.shape[0] * heightScale)
  img = cv.resize(img, (newW, newH))
  return img

def calcHist(img):
  rgb = cv.split(img)
  histSize = 256 # RGB is 0 - 255, so the size is 256 why??? <<< what is this value.
  histRange = (0, 256)
  accumulate = False # Clear histogram and make bins have same size
  b_hist = cv.calcHist(rgb, [0], None, [histSize], histRange, accumulate=accumulate)
  g_hist = cv.calcHist(rgb, [1], None, [histSize], histRange, accumulate=accumulate)
  r_hist = cv.calcHist(rgb, [2], None, [histSize], histRange, accumulate=accumulate)
  hist_w = img.shape[1]
  hist_h = img.shape[0]
  bin_w = round( hist_w/histSize ) 
  histogram = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
  cv.normalize(b_hist, b_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
  cv.normalize(r_hist, r_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
  cv.normalize(g_hist, g_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
  for i in range(1, histSize):
    cv.line(histogram, ( bin_w*(i-1), hist_h - int(np.round(b_hist[i-1])) ),
    ( bin_w*(i), hist_h - int(np.round(b_hist[i])) ),
    ( 255, 0, 0), thickness=2)
    cv.line(histogram, ( bin_w*(i-1), hist_h - int(np.round(g_hist[i-1])) ),
    ( bin_w*(i), hist_h - int(np.round(g_hist[i])) ),
    ( 0, 255, 0), thickness=2)
    cv.line(histogram, ( bin_w*(i-1), hist_h - int(np.round(r_hist[i-1])) ),
    ( bin_w*(i), hist_h - int(np.round(r_hist[i])) ),
    ( 0, 0, 255), thickness=2)
  return histogram