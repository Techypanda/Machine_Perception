import cv2 as cv
import numpy as np
import os


def main():
    files = os.listdir("./in")
    for file in files:
        if (file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg")):
            colorImg = cv.imread("./in/" + file, cv.IMREAD_COLOR)
            grayVersion = cv.cvtColor(colorImg, cv.COLOR_BGR2GRAY)
            HSVVersion = cv.cvtColor(colorImg, cv.COLOR_BGR2HSV)
            LUVVersion = cv.cvtColor(colorImg, cv.COLOR_BGR2LUV)
            LABVersion = cv.cvtColor(colorImg, cv.COLOR_BGR2LAB)
            cv.imshow("Color Version", colorImg)
            cv.imshow("Grayscale Version", grayVersion)
            cv.imshow("HSV Version", HSVVersion)
            cv.imshow("LUV Version", LUVVersion)
            cv.imshow("LAB Version", LABVersion)
            cv.waitKey()
            cv.imwrite("./out/ColorVersion{}".format(file), colorImg)
            cv.imwrite("./out/Grayscale{}".format(file), grayVersion)
            cv.imwrite("./out/HSV{}".format(file), HSVVersion)
            cv.imwrite("./out/LUV{}".format(file), LUVVersion)
            cv.imwrite("./out/LAB{}".format(file), LABVersion)

if (__name__ == "__main__"):
    main()