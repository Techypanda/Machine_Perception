import cv2 as cv
import numpy as np

def main():
  image = cv.imread("./in/prac02ex05img01.png", cv.IMREAD_GRAYSCALE)
  equalized = cv.equalizeHist(image)
  image = np.concatenate((image, equalized), 1)
  cv.imshow("Output", image)
  cv.waitKey()

if __name__ == "__main__":
  main()