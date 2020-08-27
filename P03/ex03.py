import cv2 as cv
import numpy as np
import math

image = 1
blank = 1
rho = 1
angle = np.pi/180
threshold = 200

def updateRho(val):
  global rho
  rho = val
  print("RHO IS NOW " + str(rho))
  performHough()

def updateAngle(val):
  global angle
  if val > 0:
    angle = float(val) * (np.pi/180)
  print("ANGLE IS NOW " + str(angle))
  performHough()

def updateThreshold(val):
  global threshold
  threshold = val
  print("THRESHOLD IS NOW " + str(threshold))
  performHough()

def performHough():
  global rho, angle, threshold, blank, image
  print("RHO IS NOW " + str(rho))
  print("ANGLE IS NOW " + str(angle))
  print("THRESHOLD IS NOW " + str(threshold))
  edges = cv.Canny(image, 100, 300)
  cv.imshow('edges',edges)
  lines = cv.HoughLines(edges, rho, angle, threshold)
  #lines = cv.HoughLinesP(edges,rho,angle,threshold, 5, 5)
  if lines is None:
    cv.imshow("Edge Detection", blank)
  else:
    #copy = image.copy()
    #for x1,y1,x2,y2 in lines[0]:
    #  copy = cv.line(copy,(x1,y1),(x2,y2),(0,255,0),2)
    #cv.imshow("Edge Detection", copy)
    copy = image.copy()
    for line in lines:
      d, t = line[0]
      a = np.cos(t)
      b = np.sin(t)
      x = a * d
      y = b * d
      p0 = ( int(x + 1000 * (-b)), int(y + 1000 * (a)) )
      p1 = ( int(x - 1000 * (-b)), int(y - 1000 * (a)) )
      copy = cv.line(copy, p0, p1, (0,0,255), 2)
    cv.imshow("Edge Detection", copy)

def hough(name):
  global image, blank, rho, angle, threshold
  image = cv.imread(name)
  blank = np.zeros((image.shape[0], image.shape[1]), np.uint8)
  performHough()
  cv.createTrackbar("Rho Value", "Edge Detection", rho, 10, updateRho)
  cv.createTrackbar("Angle Value", "Edge Detection", int(angle), 360, updateAngle)
  cv.createTrackbar("Threshold Value", "Edge Detection", threshold, 2000, updateThreshold)
  cv.waitKey()

def main():
  hough("./in_images/prac03ex03img02.jpg")

if __name__ == "__main__":
  main()