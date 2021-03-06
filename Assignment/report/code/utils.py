import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import math
import os

def setup():
  try:
    os.mkdir("./out_files")
    os.mkdir("./out_files/rotation")
    os.mkdir("./out_files/scale")
    os.mkdir("./out_files/task_2")
    os.mkdir("./out_files/task_2/rotation")
    os.mkdir("./out_files/task_2/scaled")
    os.mkdir("./out_files/task_3")
    os.mkdir("./out_files/task_3/card_objects")
    os.mkdir("./out_files/task_3/dugong_objects")
    os.mkdir("./out_files/task_4")
    os.mkdir("./out_files/task_4/card")
    os.mkdir("./out_files/task_4/dugong")
  except FileExistsError:
    print("Required Folders Exist")
  except:
    print("Unable to create required folders, please create them manually.")

def rAngle(img, ang): # angle in radians.
  canvas = np.zeros((int(img.shape[0] * 2), int(img.shape[1] * 2), 3), dtype=np.uint8)
  h, w, z = img.shape
  hh, ww, z = canvas.shape
  yoff = round((hh-h)/2)
  xoff = round((ww-w)/2)
  result = canvas.copy()
  result[yoff:yoff+h, xoff:xoff+w] = img
  img = result
  degrees = (ang * 180) / math.pi
  center = tuple(np.array(img.shape[1::-1]) / 2)
  rotateMatrix = cv.getRotationMatrix2D(center, degrees, 1.0)
  return cv.warpAffine(img, rotateMatrix, img.shape[1::-1], flags=cv.INTER_LINEAR)

def scaleImg(img, widthScale, heightScale):
  newW = int(img.shape[1] * widthScale)
  newH = int(img.shape[0] * heightScale)
  img = cv.resize(img, (newW, newH))
  return img

def calcHist(img, filename):
  color = ('b','g','r')
  for i, col in enumerate(color):
    histr = cv.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histr, color = col)
    plt.xlim([0, 256])
  plt.savefig(filename)
  plt.clf()

def placeOn(img, desiredWidth, desiredHeight):
  canvas = np.zeros((desiredHeight, desiredWidth), dtype=np.uint8)
  canvas[0:img.shape[0], 0:img.shape[1]] = img
  return canvas

def toFile(arr, filename):
  with open(filename, 'w') as file:
    for y in range(len(arr)):
      for x in range(len(arr[y])):
        file.write(str(arr[y][x]))
      file.write('\n')

def hogVariation(H0, H1):
  TOL = 1.0 # 0.0 - 1.0 is considered okay
  distance = cv.norm(H0 - H1)
  variant = distance >= -1 and distance <= 1.0
  if (variant):
    return """
TOLERANCE: {0}
H0 = {1}
H1 = {2}
Distance Between norm(h0 - h1) = {3}
-1.0 <= Distance <= 1.0: {4}
Transformation is invariant as distance is relatively small compared to norm of H0
    """.format(
      str(TOL), 
      str(H0), 
      str(H1), 
      str(cv.norm(H0 - H1)), 
      str(variant)
    )
  else:
    return """
TOLERANCE: {0}
H0 = {1}
H1 = {2}
Distance Between norm(h0 - h1) = {3}
-1.0 <= Distance <= 1.0: {4}
Transformation is variant as distance is not relatively small compared to norm of H0
    """.format(
      str(TOL), 
      str(H0), 
      str(H1), 
      str(cv.norm(H0 - H1)), 
      str(variant)
    )


def concon(arr):
  conflicts = {} # dict
  cc = 0
  for y in range(len(arr)):
    for x in range(len(arr[y])):
      if arr[y][x] == 1:
        if y > 0:
          if x > 0:
            if arr[y][x-1] != 0:
              if arr[y-1][x] != 0:
                if arr[y-1][x] != arr[y][x-1]:
                  if conflicts.get(arr[y-1][x]) != None: #TOP
                    conflicts[arr[y-1][x]].add(arr[y][x-1]) # add left to up
                    conflicts[arr[y-1][x]].add(arr[y-1][x]) # add up to left
                  else:
                    z = set()
                    z.add(arr[y][x-1]) # add left
                    z.add(arr[y-1][x]) # add up
                    conflicts[arr[y-1][x]] = z
                  if conflicts.get(arr[y][x-1]) != None: # LEFT
                    conflicts[arr[y][x-1]].add(arr[y-1][x]) # add up
                    conflicts[arr[y][x-1]].add(arr[y][x-1]) # add left
                  else:
                    z = set()
                    z.add(arr[y-1][x]) # add up
                    z.add(arr[y][x-1]) # add left
                    conflicts[arr[y][x-1]] = z
                  # merge the sets.
                  conflicts[arr[y-1][x]].update(conflicts[arr[y][x-1]])
                  conflicts[arr[y][x-1]].update(conflicts[arr[y-1][x]])
                  # UPDATE THEIR CHILDREN
                  for child in conflicts[arr[y-1][x]]:
                    conflicts[child].update(conflicts[arr[y-1][x]])
                  for child in conflicts[arr[y][x-1]]:
                    conflicts[child].update(conflicts[arr[y][x-1]])
                arr[y][x] = arr[y][x-1]
              else:
                arr[y][x] = arr[y][x-1]
            elif arr[y-1][x] != 0:
              arr[y][x] = arr[y-1][x]
            else: #not connected
              cc += 1
              arr[y][x] = cc
          elif arr[y-1][x] != 0:
            arr[y][x] = arr[y-1][x]
          else: # not connected
            cc += 1
            arr[y][x] = cc
        elif x > 0:
          if arr[y][x-1] != 0:
            arr[y][x] = arr[y][x-1]
          else: # not connected
            cc += 1
            arr[y][x] = cc
        else: # not connected
          cc += 1
          arr[y][x] = cc
      elif arr[y][x] == 0:
        arr[y][x] = 0
  for y in range(len(arr)):
    for x in range(len(arr[y])):
      if conflicts.get(arr[y][x]) != None:
        arr[y][x] = min(conflicts[arr[y][x]])

def conObjects(arr):
  uniqueObjs = {}
  for y in range(len(arr)):
    for x in range(len(arr[y])):
      if uniqueObjs.get(arr[y][x]) != None:
        uniqueObjs[arr[y][x]].append((y,x))
      else:
        new = []
        new.append((y,x))
        uniqueObjs[arr[y][x]] = new
  return uniqueObjs

def imgToBinary(img):
  binOutput = np.zeros(shape = (img.shape[0], img.shape[1]), dtype=np.int)
  for y in range(img.shape[0]):
    for x in range(img.shape[1]):
      if img[y, x] == 255:
        binOutput[y][x] = 0
      else:
        binOutput[y][x] = 1
  return binOutput