import cv2 as cv
import numpy as np
import utils
import math

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
  # Chosen Feature Is The Number 2 in card.
  hog = cv.HOGDescriptor('hog.xml')
  test = images['neutral'][0][5:69, 0:64]
  cv.imshow('crop', test)
  cv.waitKey()

for imageSource in imageSources:
  images = { 
    'neutral': [cv.imread(imageSource, cv.IMREAD_COLOR)],
    '75%': [utils.scaleImg(cv.imread(imageSource, cv.IMREAD_COLOR), 0.75, 0.75)],
    '50%': [utils.scaleImg(cv.imread(imageSource, cv.IMREAD_COLOR), 0.50, 0.50)],
    'widthReduced': [utils.scaleImg(cv.imread(imageSource, cv.IMREAD_COLOR), 0.50, 1)],
    'heightReduced': [utils.scaleImg(cv.imread(imageSource, cv.IMREAD_COLOR), 1, 0.50)]
  }