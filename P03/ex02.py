import cv2 as cv
import numpy as np
import Kernel

# Global Time!!! OOSE CONCEPTS OUT DA WINDOW BB
thresholdOne = 100
thresholdTwo = 200
window = "Canny Edge Detection"
#Prewit = Kernel([[-1 , 0, 1],[-1, 0, 1],[-1, 0, 1]],[[-1, -1, -1],[0, 0, 0],[1, 1, 1]])
#Sobel = Kernel([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]],[[-1, -2, -1],[0,0,0],[1,2,1]])

def thresholdChangeO(val):
    global image, thresholdOne, thresholdTwo
    thresholdOne = val
    display = cv.Canny(image, thresholdOne, thresholdTwo)
    cv.imshow(window, display)

def thresholdChangeT(val):
    global image, thresholdOne, thresholdTwo
    thresholdTwo = val
    display = cv.Canny(image, thresholdOne, thresholdTwo)
    cv.imshow(window, display)

def cannyEdgeDetection():
    global image, thresholdOne, thresholdTwo
    image = cv.imread("./in_images/prac03ex02img01.jpg", cv.IMREAD_UNCHANGED)
    Sobels = Kernel.applyKernel(image, "sobel")
    display = cv.Canny(image, thresholdOne, thresholdTwo)
    cv.imshow(window, display)
    cv.createTrackbar("Threshold One", window, thresholdOne, 1000, thresholdChangeO)
    cv.createTrackbar("Threshold Two", window, thresholdTwo, 1000, thresholdChangeT)
    cv.waitKey()

def main():
    cannyEdgeDetection()

if __name__ == "__main__":
    main()