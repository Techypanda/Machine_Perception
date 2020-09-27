import cv2 as cv
import numpy as np
import utils
utils.setup()


colorspaces = {
  "RGB": cv.COLOR_BGR2RGB, 
  "LAB": cv.COLOR_BGR2LAB,
  "XYZ": cv.COLOR_BGR2XYZ,
  "YUV": cv.COLOR_BGR2YUV,
  "HSV": cv.COLOR_BGR2HSV
}

def kmeans(image, pixel_val, eps=0.2, k = 3): # default epislon of 0.2
  pixel_val = np.float32(pixel_val)
  criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, eps)
  _, labels, (centers) = cv.kmeans(pixel_val, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
  centers = np.uint8(centers)
  labels = labels.flatten()
  segmented_image = centers[labels.flatten()]
  segmented_image = segmented_image.reshape(image.shape)
  return segmented_image

dugong = cv.imread("./images/Dugong.jpg")
diamond = cv.imread("./images/diamond2.png")
for colorspace, transform in colorspaces.items():
  for i in range(1, 4):
    img = kmeans(cv.cvtColor(diamond, transform), cv.cvtColor(diamond, transform).reshape((-1, 3)), k=i)
    cv.imwrite("./out_files/task_4/card/{} clusters {}.png".format(i, colorspace), img)
  for i in range(1, 5):
    img = kmeans(cv.cvtColor(dugong, transform), cv.cvtColor(dugong, transform).reshape((-1, 3)), k=i)
    cv.imwrite("./out_files/task_4/dugong/{} clusters {}.png".format(i, colorspace), img)