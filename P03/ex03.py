import cv2 as cv
import numpy as np

def hough(image):
  orig = cv.imread(image, cv.IMREAD_COLOR)
  edges = cv.imread(image, cv.IMREAD_GRAYSCALE)
  edges = cv.Canny(edges, 100, 200)
  coords = [(x,y) for y in range(0, edges.shape[0]) for x in range(0, edges.shape[1]) if (edges[y][x] != (0,0,0)).any()]
  for coord in coords:
    orig = cv.circle(
      orig,
      coord,
      1,
      (0,0,255))
  cv.imshow("Edge Detection", edges)
  cv.imshow("ORIGINAL", orig)
  cv.waitKey()

def main():
  hough("./in_images/image_01.png")

if __name__ == "__main__":
  main()