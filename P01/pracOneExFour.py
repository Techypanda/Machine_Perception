import cv2 as cv
import numpy as np
import math

def main():
    image = cv.imread("./in_images/prac01ex04img01.png", cv.IMREAD_COLOR)
    rows, cols, color = image.shape
    p1, p2, p0 = 0, 0, (0,0)
    # FIND HORIZONTAL POINT ONE.
    for i in range(0,image.shape[1]):
        if (image[i,0] != np.array([0,0,0])).all():
            p1 = (i, 0)
    # FIND VERTICAL
    for i in range(0, image.shape[0]):
        if (image[0,i] != np.array([0,0,0])).all():
            p2 = (0, i)

    d = int(round(math.degrees(math.atan(p1[0] / p2[1]))))
    transformMatrix = cv.getRotationMatrix2D((rows/2, cols/2), d, 1)
    rotatedImage = cv.warpAffine(image, transformMatrix, (cols, rows))
    cv.imshow('Rotated', rotatedImage)
    cv.waitKey()
    print("Writing To File!")
    cv.imwrite("./out_images/prac01ex04rotated.png", rotatedImage)

if (__name__ == "__main__"):
    main()