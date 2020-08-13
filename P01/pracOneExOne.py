import cv2 as cv
import os
import numpy as np

def main():
    files = os.listdir("./in_images")
    for file in files:
        if (file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg")):
            image = cv.imread("./in_images/" + file, cv.IMREAD_COLOR)
            dimensions = image.shape[0], image.shape[1]
            print("{}: \n   Height: {}px\n   Width: {}px".format(file, dimensions[0], dimensions[1]))
            rgb = cv.split(image)
            histSize = 256 # RGB is 0 - 255, so the size is 256 why??? <<< what is this value.
            histRange = (0, 256)
            accumulate = False # Clear histogram and make bins have same size
            b_hist = cv.calcHist(rgb, [0], None, [histSize], histRange, accumulate=accumulate)
            g_hist = cv.calcHist(rgb, [1], None, [histSize], histRange, accumulate=accumulate)
            r_hist = cv.calcHist(rgb, [2], None, [histSize], histRange, accumulate=accumulate)
            hist_w = 1280
            hist_h = 720
            bin_w = round( hist_w/histSize ) 
            histogram = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
            cv.normalize(b_hist, b_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
            cv.normalize(r_hist, r_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
            cv.normalize(g_hist, g_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
            for i in range(1, histSize):
                cv.line(histogram, ( bin_w*(i-1), hist_h - int(np.round(b_hist[i-1])) ),
                ( bin_w*(i), hist_h - int(np.round(b_hist[i])) ),
                ( 255, 0, 0), thickness=2)
                cv.line(histogram, ( bin_w*(i-1), hist_h - int(np.round(g_hist[i-1])) ),
                ( bin_w*(i), hist_h - int(np.round(g_hist[i])) ),
                ( 0, 255, 0), thickness=2)
                cv.line(histogram, ( bin_w*(i-1), hist_h - int(np.round(r_hist[i-1])) ),
                ( bin_w*(i), hist_h - int(np.round(r_hist[i])) ),
                ( 0, 0, 255), thickness=2)
            cv.imshow('Source Image', image)
            cv.imshow('Histogram Image', histogram)
            cv.waitKey()
            cv.destroyAllWindows()
            print("Now Resizing And Outputting To out_images as 50%")
            resizedImage = cv.resize(image, (int(round(dimensions[1] / 2)), int(round(dimensions[0] / 2))))
            cv.imwrite("./out_images/" + file, resizedImage)
            window1 = cv.imshow('Original Image', image)
            window2 = cv.imshow('Scaled Image', resizedImage)
            cv.waitKey()
            cv.destroyAllWindows()

if (__name__ == "__main__"):
    main()