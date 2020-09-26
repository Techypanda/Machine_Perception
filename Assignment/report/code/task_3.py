import cv2 as cv
import numpy as np
import utils
import random

card = cv.imread("./images/diamond2.png", cv.IMREAD_GRAYSCALE)
dugong = cv.imread('./images/Dugong.jpg', cv.IMREAD_GRAYSCALE)
blockSize = 11
inbuiltBlur = 5
dugong = cv.adaptiveThreshold(dugong,255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
cv.THRESH_BINARY, blockSize, inbuiltBlur)
ker = np.ones((10,10), np.uint8)
dugong = cv.morphologyEx(dugong, cv.MORPH_OPEN, ker)
ker = np.ones((4,5), np.uint8)
dugong = cv.morphologyEx(dugong, cv.MORPH_CLOSE, ker)
_, card = cv.threshold(card, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
dugong = cv.bitwise_not(dugong)
card = cv.bitwise_not(card)
cv.imwrite("./out_files/task_3/dugong_objects/BinaryImage.png", dugong)
cv.imwrite("./out_files/task_3/card_objects/BinaryImage.png", card)
dugong = cv.bitwise_not(dugong)
card = cv.bitwise_not(card)
dugong = utils.imgToBinary(dugong)
card = utils.imgToBinary(card)
utils.toFile(dugong, './out_files/task_3/dugong_objects/dugong.txt')
utils.toFile(card, './out_files/task_3/card_objects/card_before.txt')

utils.concon(dugong); utils.concon(card) # my connected components method

utils.toFile(dugong, './out_files/task_3/dugong_objects/dugong_objects.txt')
utils.toFile(card, './out_files/task_3/card_objects/card_objects.txt')
dugongObjs = utils.conObjects(dugong)
cardObjs = utils.conObjects(card)
objs = [(dugongObjs, './images/Dugong.jpg', 
'./out_files/task_3/dugong_objects/detectedObjectsCCL.png', 
'./out_files/task_3/dugong_objects/'),
(cardObjs, './images/diamond2.png', 
'./out_files/task_3/card_objects/detectedObjectsCCL.png', 
'./out_files/task_3/card_objects/')]
for obj in objs:
  clr = cv.imread(obj[1], cv.IMREAD_COLOR)
  for key, coords in obj[0].items():
    color =  (random.randint(0, 255), random.randint(0, 255), 
    random.randint(0, 255))
    for cord in coords:
      clr = cv.rectangle(clr, 
      (cord[1], cord[0]), (cord[1], cord[0]), color)
  cv.imwrite(obj[2], clr)

  # Extract The Objects Into Seperate Files
  objectNo = 1
  for key, coords in obj[0].items():
    xMin = coords[0][1]; xMax = coords[0][1]
    yMin = coords[0][0]; yMax = coords[0][0]
    for coord in coords:
      if coord[1] < xMin:
        xMin = coord[1]
      elif coord[1] > xMax:
        xMax = coord[1]
      if coord[0] < yMin:
        yMin = coord[0]
      elif coord[0] > yMax:
        yMax = coord[0]
    out = cv.imread(obj[1], cv.IMREAD_COLOR)
    out = out[yMin:yMax, xMin:xMax]
    cv.imwrite("{}DetectedObject-{}.png".format(obj[3], objectNo), out)
    objectNo += 1