import cv2 as cv
import numpy as np

def MSER(image):
  image = cv.imread(image, cv.IMREAD_GRAYSCALE)
  vis = image.copy()
  MSER = cv.MSER_create()
  regions = MSER.detectRegions(image)
  hulls = [cv.convexHull(p.reshape(-1, 1, 2)) for p in regions[0]]
  vis = cv.cvtColor(vis, cv.COLOR_GRAY2RGB)
  cv.polylines(vis, hulls, 1, (0, 255, 0))
  cv.imshow("Image", vis)
  cv.waitKey()

def main():
  MSER("./in_images/prac03ex04img02.png")

if __name__ == "__main__":
  main()