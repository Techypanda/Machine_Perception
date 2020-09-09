import cv2 as cv;
import numpy as np;

image = cv.imread("in_images/prac04ex01img01.png")
imageT = cv.imread("in_images/prac04ex01img02.png")
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(imageT, cv.COLOR_BGR2GRAY)

SIFT = cv.xfeatures2d.SIFT_create()
kp1, des1 = SIFT.detectAndCompute(gray, None)
kp2, des2 = SIFT.detectAndCompute(gray2, None)

bf = cv.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2) # k = 2 is the only value that works. ._.

good = []
for m,n in matches:
  if m.distance < 0.75*n.distance: # The higher the constant more picked up.
    good.append([m])

matches = cv.drawMatchesKnn(image, kp1, imageT, kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv.imshow("Da Matches", matches)
cv.waitKey()

#image = np.concatenate((image, imageT), 1)

cv.imshow("Sift Keypoints", image)
cv.waitKey()


#kp, des = SIFT.compute(gray, kp)
#print(des[0])
#cv.imshow('ahh', des)
#cv.waitKey()