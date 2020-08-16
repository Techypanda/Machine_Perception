import cv2 as cv
import numpy as np
import sys

WINDOW_TITLE = "Pixel Transformation"
alpha = 0 # Guess Machine Perception is where i start using globals huh...
beta = 0
image = 0

def betaChange(val):
  global alpha, beta
  beta = val
  pixelTransformation()

def alphaChange(val):
  global alpha, beta
  alpha = val
  pixelTransformation()

def pixelTransformation():
  global alpha, beta, image
  print("{} {}".format(alpha, beta))
  copy = image.copy()
  for x in range(0, copy.shape[0]):
    for y in range(0, copy.shape[1]):
      BGR = copy[x,y] # Blue green red values
      for i in range(0, 3): # R G B
        BGR[i] = np.clip((alpha * BGR[i]) + beta, 0, 255)
      copy[x,y] = BGR
  cv.imshow(WINDOW_TITLE, copy)

def main():
  global image
  image = cv.imread("./in/prac02ex04img01.png", cv.IMREAD_COLOR)
  global alpha, beta
  cv.imshow('Original', image)
  cv.imshow(WINDOW_TITLE, image)
  cv.createTrackbar("Alpha Value", WINDOW_TITLE, beta, 10, alphaChange)
  cv.createTrackbar("Beta Value", WINDOW_TITLE, alpha, 10, betaChange)
  cv.waitKey()

if __name__ == "__main__":
  main()