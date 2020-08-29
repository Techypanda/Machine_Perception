import cv2 as cv
image = "./in_images/image_02.jpg"
incorrect = cv.imread(image, cv.COLOR_BGR2GRAY)
correct = cv.imread(image, cv.IMREAD_GRAYSCALE)
cv.imshow("Correct", correct)
cv.imshow("Incorrect", incorrect)
cv.waitKey()