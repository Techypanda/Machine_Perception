import cv2 as cv
import numpy as np

def open(src, kernel):
  return cv.dilate(cv.erode(src, kernel), kernel)

def close(src, kernel):
  return cv.erode(cv.dilate(src,kernel), kernel)

def morphgradient(src, kernel):
  return cv.dilate(src, kernel) - cv.erode(src, kernel)

def blackhat(src, kernel):
  return close(src, kernel) - src

def comparison(image, operation):
  kernelRect = cv.getStructuringElement(cv.MORPH_RECT, (5,5))
  kernelCross = cv.getStructuringElement(cv.MORPH_CROSS, (5,5))
  kernelEllipse = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
  openedRect = operation(image, kernelRect)
  openedCross = operation(image, kernelCross)
  openedEllipse = operation(image, kernelEllipse)
  cv.imshow("Original", image)
  cv.imshow("Rect Kernel", openedRect)
  cv.imshow("Cross Kernel", openedCross)
  cv.imshow("Ellipse Kernel", openedEllipse)

def main():
  image = cv.imread("./in/prac02ex06img02.png", cv.IMREAD_COLOR)
  comparison(image, open)
  cv.waitKey()
  cv.destroyAllWindows()
  comparison(image, close)
  cv.waitKey()
  cv.destroyAllWindows()
  comparison(image, morphgradient)
  cv.waitKey()
  cv.destroyAllWindows()
  comparison(image, blackhat)
  cv.waitKey()


if __name__ == "__main__":
  main()