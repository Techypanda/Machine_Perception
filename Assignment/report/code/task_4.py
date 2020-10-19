import cv2 as cv
import numpy as np
import utils
utils.setup()


colorspaces = {
  "RGB": (cv.COLOR_BGR2RGB, cv.COLOR_RGB2BGR), 
  "LAB": (cv.COLOR_BGR2LAB, cv.COLOR_LAB2BGR),
  "XYZ": (cv.COLOR_BGR2XYZ, cv.COLOR_XYZ2BGR),
  "YUV": (cv.COLOR_BGR2YUV, cv.COLOR_YUV2BGR),
  "HSV": (cv.COLOR_BGR2HSV, cv.COLOR_HSV2BGR)
}
def kmeans(image, transformBack, k = 4):
  Z = np.float32(image.reshape((-1,3)))
  criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
  _,labels,centers = cv.kmeans(Z, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
  labels = labels.reshape((image.shape[:-1]))
  reduced = np.uint8(centers)[labels]
  reduced = cv.cvtColor(reduced, transformBack)
  image = cv.cvtColor(image, transformBack)
  result = [np.hstack(
    [image, reduced]
  )]
  for i, c in enumerate(centers):
    mask = cv.inRange(labels, i, i)
    mask = np.dstack([mask]*3) # Make it 3 channel
    masked_image = cv.bitwise_and(image, mask)
    masked_reduced_image = cv.bitwise_and(reduced, mask)
    result.append(np.hstack([masked_image, masked_reduced_image]))
  return np.vstack(result)


dugong = cv.imread("./images/Dugong.jpg")
diamond = cv.imread("./images/diamond2.png")

for colorspace, transform in colorspaces.items():
  for i in range(1, 4):
    img = kmeans(cv.cvtColor(diamond, transform[0]), transform[1], k=i)
    cv.imwrite("./out_files/task_4/card/{} clusters {}.png".format(i, colorspace), img)
  for i in range(1, 5):
    img = kmeans(cv.cvtColor(dugong, transform[0]), transform[1], k=i)
    cv.imwrite("./out_files/task_4/dugong/{} clusters {}.png".format(i, colorspace), img)