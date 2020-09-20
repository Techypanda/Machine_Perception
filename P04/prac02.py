import cv2 as cv
import numpy as np
import extraction as ext
import random

img = cv.imread('./in_images/prac04ex02img01.png', cv.IMREAD_GRAYSCALE)
clr = cv.imread('./in_images/prac04ex02img01.png', cv.IMREAD_COLOR)

# Otsu's thresholding
otsRet, thr = cv.threshold(img,0,255, cv.THRESH_BINARY + cv.THRESH_OTSU)
cv.imshow('orig', img)
cv.imshow('otsu', thr)

binOutput = np.zeros(shape = (thr.shape[0], thr.shape[1]), dtype=np.int)

for y in range(thr.shape[0]):
  for x in range(thr.shape[1]):
    if thr[y, x] == 255:
      binOutput[y][x] = 0
    else:
      binOutput[y][x] = 1

ext.toFile(binOutput, 'binaryBefore.txt')
ext.concon(binOutput)

ext.toFile(binOutput, 'binary.txt')
uni = ext.conObjects(binOutput)

for key, coords in uni.items():
  color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
  for cord in coords:
    clr = cv.rectangle(clr, (cord[1], cord[0]), (cord[1], cord[0]), color)

cv.imshow('color', clr)
'''cc = 0
  for x in range(len(binOutput)):
    for y in range(len(binOutput[0])):
      if binOutput[x][y] == 1:
        if x > 0 and binOutput[x-1][y] != 0: # or binOutput[x][y-1] != 0: # connected
          if y > 0 and binOutput[x][y-1] != 0:
            print(str(binOutput[x-1][y]) + ' == ' + str(binOutput[x][y-1]))
            binOutput[x][y] = binOutput[x-1][y]
          else:
            binOutput[x][y] = binOutput[x-1][y]
        else:
          cc += 1
          binOutput[x][y] = cc
      else:
        binOutput[x][y] = 0'''

cv.waitKey()
cv.destroyAllWindows()