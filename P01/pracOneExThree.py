import numpy as np
import cv2 as cv

def main():
    coordinates = []
    f = open("prac01ex02crop.txt")
    for string in f.read().split(" "):
        coordinates.append(int(string))
    coordinates = np.array(coordinates)
    image = cv.imread("./in_images/prac01ex02img01.png", cv.IMREAD_COLOR)
    image = cv.rectangle(image, (coordinates[0], coordinates[1]), (coordinates[2], coordinates[3]), (0,0,255), 2)
    image = cv.circle(image, (coordinates[0], coordinates[1]), 2, (0,255,0), 2)
    image = cv.circle(image, (coordinates[0], coordinates[3]), 2, (0,255,0), 2)
    image = cv.circle(image, (coordinates[2], coordinates[1]), 2, (0,255,0), 2)
    image = cv.circle(image, (coordinates[2], coordinates[3]), 2, (0,255,0), 2)
    cv.imshow('Show', image)
    cv.waitKey()
    print("Writing to file!")
    cv.imwrite("./out_images/carwithdarectangle.png", image)
    
if (__name__ == "__main__"):
    main()