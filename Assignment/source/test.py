import cv2 
import utils
img = cv2.imread('./images/Histogram_Calculation_Original_Image.jpg', cv2.IMREAD_COLOR)
cv2.imshow('test', utils.calcHist(img))
cv2.imshow('o', img)
cv2.waitKey()