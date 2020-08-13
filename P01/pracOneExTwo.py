import numpy as np
import cv2 as cv

def main():
    coordinates = []
    f = open("prac01ex02crop.txt")
    for string in f.read().split(" "):
        coordinates.append(int(string))
    coordinates = np.array(coordinates)
    image = cv.imread("./in_images/prac01ex02img01.png", cv.IMREAD_COLOR)
    cropped = image[coordinates[1]:coordinates[3], coordinates[0]:coordinates[2]]
    cv.imwrite("./out_images/prac01ex02crop.png", cropped)
    cv.imshow('Source', image)
    cv.imshow('Cropped', cropped)
    cv.waitKey()

if (__name__ == "__main__"):
    main()