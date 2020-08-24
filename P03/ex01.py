import cv2 as cv
import numpy as np

# R Very large = strong corner
# | R | small = flat like :eyes: ykwbb
# R < 0 and large magnitude = edge
def harrisCorner():
  print("mmmm cump")
  for i in range(1, 4):
    image = cv.imread("./in_images/prac03ex01img0{}.png".format(i))
    print(image)

def main():
  harrisCorner()

if __name__ == "__main__":
  main()