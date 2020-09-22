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
  for image in images.values():
    image.append(utils.calcHist(image[0]))
  cv.imwrite('./out_files/rotation/{0}Neutral Hist.png'.format(title[imageSource]), np.concatenate((images['neutral'][0], images['neutral'][1]), axis=1))
  cv.imwrite('./out_files/rotation/{0}Deg 45 Hist.png'.format(title[imageSource]), np.concatenate((images['deg45'][0], images['deg45'][1]), axis=1))
  cv.imwrite('./out_files/rotation/{0}Deg -45 Hist.png'.format(title[imageSource]), np.concatenate((images['degn45'][0], images['degn45'][1]), axis=1))
  cv.imwrite('./out_files/rotation/{0}Deg 90 Hist.png'.format(title[imageSource]), np.concatenate((images['deg90'][0], images['deg90'][1]), axis=1))
  cv.imwrite('./out_files/rotation/{0}Deg -90 Hist.png'.format(title[imageSource]), np.concatenate((images['degn90'][0], images['degn90'][1]), axis=1))
  cv.imwrite('./out_files/rotation/{0}180 Deg Hist.png'.format(title[imageSource]), np.concatenate((images['flipped'][0], images['flipped'][1]), axis=1)) # Rotationally Variant
  for image in images.values():
    cpy = image[0].copy()
    cpy[cv.cornerHarris(cv.cvtColor(image[0], cv.COLOR_BGR2GRAY), 5, 9, 0.05)>0.01*cpy.max()]=[0,0,255]
    image.append(cpy)
  cv.imwrite('./out_files/rotation/{0}Neutral Corners.png'.format(title[imageSource]), images['neutral'][2])
  cv.imwrite('./out_files/rotation/{0}Deg 45 Corners.png'.format(title[imageSource]), images['deg45'][2])
  cv.imwrite('./out_files/rotation/{0}Deg -45 Corners.png'.format(title[imageSource]), images['degn45'][2])
  cv.imwrite('./out_files/rotation/{0}Deg 90 Corners.png'.format(title[imageSource]), images['deg90'][2])
  cv.imwrite('./out_files/rotation/{0}Deg -90 Corners.png'.format(title[imageSource]), images['degn90'][2])
  cv.imwrite('./out_files/rotation/{0}180 Deg Corners.png'.format(title[imageSource]), images['flipped'][2]) # Rotationally Invariant

for imageSource in imageSources: # Scaling
  images = { 
    'neutral': [cv.imread(imageSource, cv.IMREAD_COLOR)],
    '75%': [utils.scaleImg(cv.imread(imageSource, cv.IMREAD_COLOR), 0.75, 0.75)],
    '50%': [utils.scaleImg(cv.imread(imageSource, cv.IMREAD_COLOR), 0.50, 0.50)],
    'widthReduced': [utils.scaleImg(cv.imread(imageSource, cv.IMREAD_COLOR), 0.50, 1)],
    'heightReduced': [utils.scaleImg(cv.imread(imageSource, cv.IMREAD_COLOR), 1, 0.50)]
  }

  # Hist
  for image in images.values():
    image.append(utils.calcHist(image[0]))
  cv.imwrite('./out_files/scale/{0}Neutral Hist.png'.format(title[imageSource]), np.concatenate((images['neutral'][0], images['neutral'][1]), axis=1)) 
  cv.imwrite('./out_files/scale/{0}75 Hist.png'.format(title[imageSource]), np.concatenate((images['75%'][0], images['75%'][1]), axis=1))
  cv.imwrite('./out_files/scale/{0}50 Hist.png'.format(title[imageSource]), np.concatenate((images['50%'][0], images['50%'][1]), axis=1))
  cv.imwrite('./out_files/scale/{0}widthReduced Hist.png'.format(title[imageSource]), np.concatenate((images['widthReduced'][0], images['widthReduced'][1]), axis=1))
  cv.imwrite('./out_files/scale/{0}heightReduced Hist.png'.format(title[imageSource]), np.concatenate((images['heightReduced'][0], images['heightReduced'][1]), axis=1)) # Somewhat Scale Invariant
  for image in images.values():
    cpy = image[0].copy()
    cpy[cv.cornerHarris(cv.cvtColor(image[0], cv.COLOR_BGR2GRAY), 5, 9, 0.05)>0.01*cpy.max()]=[0,0,255]
    image.append(cpy)
  cv.imwrite('./out_files/scale/{0}Neutral Corners.png'.format(title[imageSource]), images['neutral'][2])
  cv.imwrite('./out_files/scale/{0}75 Corners.png'.format(title[imageSource]), images['75%'][2])
  cv.imwrite('./out_files/scale/{0}50 Corners.png'.format(title[imageSource]), images['50%'][2])
  cv.imwrite('./out_files/scale/{0}widthReduced Corners.png'.format(title[imageSource]), images['widthReduced'][2])
  cv.imwrite('./out_files/scale/{0}heightReduced Corners.png'.format(title[imageSource]), images['heightReduced'][2]) # Scale Invariant