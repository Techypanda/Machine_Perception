import cv2 as cv
import numpy as np
import utils
import math
import random
title = {'./images/diamond2.png': 'diamond', './images/Dugong.jpg': 'Dugong'} # lazy quick fix...
imageSources = ['./images/diamond2.png', './images/Dugong.jpg']
for imageSource in imageSources: # ROTATION
  images = { 
    'neutral': [cv.imread(imageSource, cv.IMREAD_COLOR)],
    'deg45': [utils.rAngle(cv.imread(imageSource, cv.IMREAD_COLOR), math.radians(45))],
    'degn45': [utils.rAngle(cv.imread(imageSource, cv.IMREAD_COLOR), math.radians(-45))],
    'deg90': [cv.rotate(cv.imread(imageSource, cv.IMREAD_COLOR), cv.ROTATE_90_CLOCKWISE)],
    'degn90': [cv.rotate(cv.imread(imageSource, cv.IMREAD_COLOR), cv.ROTATE_90_COUNTERCLOCKWISE)],
    'flipped': [cv.flip(cv.imread(imageSource, cv.IMREAD_COLOR), 0)]
  }
  for key, image in images.items(): # Hist
    utils.calcHist(image[0], './out_files/rotation/{0}{1} Hist.png'.format(title[imageSource], key))
  for image in images.values(): # Corner Detection
    cpy = image[0].copy()
    cpy[cv.cornerHarris(cv.cvtColor(image[0], cv.COLOR_BGR2GRAY), 5, 9, 0.05)>0.01*cpy.max()]=[255,0,0]
    image.append(cpy)
  cv.imwrite('./out_files/rotation/{0}Neutral Corners.png'.format(title[imageSource]), images['neutral'][1])
  cv.imwrite('./out_files/rotation/{0}Deg 45 Corners.png'.format(title[imageSource]), images['deg45'][1])
  cv.imwrite('./out_files/rotation/{0}Deg -45 Corners.png'.format(title[imageSource]), images['degn45'][1])
  cv.imwrite('./out_files/rotation/{0}Deg 90 Corners.png'.format(title[imageSource]), images['deg90'][1])
  cv.imwrite('./out_files/rotation/{0}Deg -90 Corners.png'.format(title[imageSource]), images['degn90'][1])
  cv.imwrite('./out_files/rotation/{0}180 Deg Corners.png'.format(title[imageSource]), images['flipped'][1]) # Rotationally Invariant
  for image in images.values(): # SIFT
    SIFT = cv.xfeatures2d.SIFT_create()
    kp, des = SIFT.detectAndCompute(cv.cvtColor(image[0], cv.COLOR_BGR2GRAY), None)
    image.append(cv.drawKeypoints(cv.cvtColor(image[0], cv.COLOR_BGR2GRAY), kp, image[0]))
  cv.imwrite('./out_files/rotation/{0}Neutral SIFT.png'.format(title[imageSource]), images['neutral'][2])
  cv.imwrite('./out_files/rotation/{0}Deg 45 SIFT.png'.format(title[imageSource]), images['deg45'][2])
  cv.imwrite('./out_files/rotation/{0}Deg -45 SIFT.png'.format(title[imageSource]), images['degn45'][2])
  cv.imwrite('./out_files/rotation/{0}Deg 90 SIFT.png'.format(title[imageSource]), images['deg90'][2])
  cv.imwrite('./out_files/rotation/{0}Deg -90 SIFT.png'.format(title[imageSource]), images['degn90'][2])
  cv.imwrite('./out_files/rotation/{0}180 Deg SIFT.png'.format(title[imageSource]), images['flipped'][2]) # Rotationally Invariant

for imageSource in imageSources: # Scaling
  images = { 
    'neutral': [cv.imread(imageSource, cv.IMREAD_COLOR)],
    '75%': [utils.scaleImg(cv.imread(imageSource, cv.IMREAD_COLOR), 0.75, 0.75)],
    '50%': [utils.scaleImg(cv.imread(imageSource, cv.IMREAD_COLOR), 0.50, 0.50)],
    'widthReduced': [utils.scaleImg(cv.imread(imageSource, cv.IMREAD_COLOR), 0.50, 1)],
    'heightReduced': [utils.scaleImg(cv.imread(imageSource, cv.IMREAD_COLOR), 1, 0.50)]
  }

  # Hist
  for key, image in images.items():
    utils.calcHist(image[0], './out_files/scale/{0}{1} Hist.png'.format(title[imageSource], key))
  for image in images.values():
    cpy = image[0].copy()
    cpy[cv.cornerHarris(cv.cvtColor(image[0], cv.COLOR_BGR2GRAY), 5, 9, 0.05)>0.01*cpy.max()]=[255,0,0]
    image.append(cpy)
  cv.imwrite('./out_files/scale/{0}Neutral Corners.png'.format(title[imageSource]), images['neutral'][1])
  cv.imwrite('./out_files/scale/{0}75 Corners.png'.format(title[imageSource]), images['75%'][1])
  cv.imwrite('./out_files/scale/{0}50 Corners.png'.format(title[imageSource]), images['50%'][1])
  cv.imwrite('./out_files/scale/{0}widthReduced Corners.png'.format(title[imageSource]), images['widthReduced'][1])
  cv.imwrite('./out_files/scale/{0}heightReduced Corners.png'.format(title[imageSource]), images['heightReduced'][1]) # Scale Invariant
  for image in images.values(): # SIFT
    SIFT = cv.xfeatures2d.SIFT_create()
    kp, des = SIFT.detectAndCompute(cv.cvtColor(image[0], cv.COLOR_BGR2GRAY), None)
    image.append(cv.drawKeypoints(cv.cvtColor(image[0], cv.COLOR_BGR2GRAY), kp, image[0]))
  cv.imwrite('./out_files/scale/{0}Neutral SIFT.png'.format(title[imageSource]), images['neutral'][2])
  cv.imwrite('./out_files/scale/{0}75 SIFT.png'.format(title[imageSource]), images['75%'][2])
  cv.imwrite('./out_files/scale/{0}50 SIFT.png'.format(title[imageSource]), images['50%'][2])
  cv.imwrite('./out_files/scale/{0}widthReduced SIFT.png'.format(title[imageSource]), images['widthReduced'][2])
  cv.imwrite('./out_files/scale/{0}heightReduced SIFT.png'.format(title[imageSource]), images['heightReduced'][2]) # Mostly Scale Invariant