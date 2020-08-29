import cv2 as cv
import numpy as np
import sys

def main():
    file = sys.argv[1]
    image = cv.imread(file, 0)
    cv.imshow("Initial Load", image);
    filter = cv.Canny(image, 50, 150, 3)
    hough_lines = cv.HoughLinesP(filter, 1, np.pi/180, 100, 100, 10)
    for line in hough_lines:
        x1,y1,x2,y2 = line[0]
        cv.line(image, (x1,y1), (x2,y2), (0,255,255), 2)
        
    cv.imshow("Lines", image)
    cv.waitKey() #sugma
    
if __name__ == "__main__":
    main()