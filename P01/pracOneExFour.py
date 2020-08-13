import cv2 as cv
import numpy as np

def main():
    image = cv.imread("./in_images/prac01ex04img01.png", cv.IMREAD_COLOR)
    rows, cols, color = image.shape
    transformMatrix = cv.getRotationMatrix2D((rows/2, cols/2), 45, 1)
    rotatedImage = cv.warpAffine(image, transformMatrix, (cols, rows))
    cv.imshow('Unrotated', image)
    cv.imshow('Rotated', rotatedImage)
    cv.waitKey()
    print("Writing To File!")
    cv.imwrite("./out_images/prac01ex04rotated.png", rotatedImage)

if (__name__ == "__main__"):
    main()