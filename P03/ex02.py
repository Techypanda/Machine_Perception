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
    '''
    Accumulator Array: A 2D Table where x is r, and y is theta, put the values in here, poll how many are same and that changes the intensity

    For each pixel in edges detected (Canny Output) you need to draw a mathematical line through this pixel at every theta i.e (0 - 360) and record the shortest distance of origin to that point (r), record the r and theta for each pixel in the accumulator array.

    You can then visualise the Accumulator Array based off the intensity.
    '''
    display = cv.Canny(image, thresholdOne, thresholdTwo)
    cv.imshow(window, display)
    cv.createTrackbar("Threshold One", window, thresholdOne, 1000, thresholdChangeO)
    cv.createTrackbar("Threshold Two", window, thresholdTwo, 1000, thresholdChangeT)
    cv.waitKey()

def main():
    cannyEdgeDetection()

if __name__ == "__main__":
    main()