import cv2 as cv
import numpy as np
import sys
WINDOW_TITLE = "Median Blur"
ITERATION_MAX = 31
IMAGE_PATH = "./in/prac02ex03img01.jpg"

def slider(val):
  sliderVal = int(round(val)) if (int(round(val)) % 2 == 1) and int(round(val)) > 0 else int(round(val)) + 1
  image = cv.imread(IMAGE_PATH)
  blurred = cv.medianBlur(image, sliderVal)
  cv.imshow(WINDOW_TITLE, blurred)

def main(image):
  print("Begin Median Filter On {}".format(image))
  image = cv.imread(IMAGE_PATH)
  if image.data:
    blurred = cv.medianBlur(image, 1)
    cv.imshow(WINDOW_TITLE, blurred)
    cv.createTrackbar('How Many Filters Applied', WINDOW_TITLE, 0, ITERATION_MAX, slider)
    cv.waitKey()
    cv.imshow(WINDOW_TITLE, blurred)
    cv.waitKey()
  else:
    print("Couldnt Open {}".format(image))

if __name__ == "__main__":
  if len(sys.argv) < 2:
    print("Requires Path To Image")
  else:
    main(sys.argv[1])