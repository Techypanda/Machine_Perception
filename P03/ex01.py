import cv2 as cv
import numpy as np

# R Very large = strong corner
# | R | small = flat like :eyes: ykwbb
# R < 0 and large magnitude = edge
def harrisCorner():
  print("Enter Kernel Size For Sobel: (ODD NUMBER)")
  ksize = int(input())
  k = 0.05 # Constant In Range [0.04 - 0.06]
  for i in range(1, 4):
    #image = cv.imread("./in_images/prac03ex01img0{}.png".format(i), cv.IMREAD_GRAYSCALE)
    image = cv.imread("./in_images/diamond2.png", cv.IMREAD_GRAYSCALE)
    cornerDetected = cv.cornerHarris(image, 5, ksize, k)
    cv.imshow("Original", image)
    cv.imshow("Harris Corner Detection", cornerDetected)
    cv.waitKey()
    # NOTES ON HARRIS DETECTION.
    # Changing K makes very little difference (I can't find it.)
    # Changing the block size changes the actual blocks placed on the corner's size.
    # Changing KSize affects how much the image is affected by sobel operator resulting in more noise being picked up

def shiTomasi():
  for i in range(1, 4):
    image = cv.imread("./in_images/prac03ex01img0{}.png".format(i), cv.IMREAD_GRAYSCALE)
    corners = cv.goodFeaturesToTrack(image, 16, 0.01, 50)
    modifiedImage = image.copy()
    currCorner = 0
    modifiedImage = cv.cvtColor(modifiedImage, cv.COLOR_GRAY2RGB)
    for corner in corners:
      modifiedImage = cv.circle(modifiedImage, (corners[currCorner, 0, 0], corners[currCorner, 0, 1]), 5, (0, 0, 255))
      currCorner += 1
    cv.imshow("Original", image)
    cv.imshow("Corners", modifiedImage)
    cv.waitKey()
    # NOTES ON SHI TOMASI
    # Much Better overall
    # Changing count to a reasonable amount makes program more accurate.
    # Changing Quality Unsure.
    # Changing Distance will effect how close the points can be.

def main():
  print("1 - Harris\n2 - ShiTomasi\nAny other number will just crash.")
  choice = int(input())
  if choice == 1:
    harrisCorner()
  elif choice == 2:
    shiTomasi()

if __name__ == "__main__":
  main()