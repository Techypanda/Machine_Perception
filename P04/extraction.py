import numpy as np

def toFile(arr, filename):
  with open(filename, 'w') as file:
    for y in range(len(arr)):
      for x in range(len(arr[y])):
        file.write(str(arr[y][x]))
      file.write('\n')

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
                    conflicts[arr[y-1][x]].add(arr[y][x-1]) # add left
                    conflicts[arr[y-1][x]].add(arr[y-1][x]) # add up
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