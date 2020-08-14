import cv2 as cv
import numpy as np
import sys

class Kernel:
    def __init__(self, *args):
        self.kernels = []
        for kernel in args:
            kernel = np.array(kernel, np.float32)
            #kernel = np.flip(kernel, 0) # Flip vertically
            #kernel = np.flip(kernel, 1) # Flip Horizontally DOESNT WORK... TRANSPOSE TIME T_T
            kernel = np.transpose(kernel)
            self.kernels.append(kernel)
    def __str__(self):
        count = 1
        outStr = ""
        for kernel in self.kernels:
            outStr += "Kernel {}: \n{}\n".format(count, kernel)
        return outStr

def presentFilter(image, kernel, timesApplied, filterTitle):
    for npArray in kernel.kernels:
        presented = 1
        currentImage = cv.filter2D(image, -1, npArray)
        cv.imshow(filterTitle, currentImage)
        cv.waitKey()
        cv.destroyAllWindows()
        while presented != timesApplied:
            presented += 1 # x++ doesnt exist in python... wtf. GROSS!
            currentImage = cv.filter2D(currentImage, -1, npArray)
            cv.imshow(filterTitle, currentImage)
            cv.waitKey()
            cv.destroyAllWindows()
            

def visualise(image):
    image = cv.imread(image, cv.IMREAD_GRAYSCALE)
    Prewit = Kernel([[-1 , 0, 1],[-1, 0, 1],[-1, 0, 1]],[[-1, -1, -1],[0, 0, 0],[1, 1, 1]])
    Sobel = Kernel([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]],[[-1, -2, -1],[0,0,0],[1,2,1]])
    Laplacian = Kernel([[0, 1, 0],[1, -4, 1],[0, 1, 0]])
    presentFilter(image, Prewit, 10, 'Prewits Kernels')
    presentFilter(image, Sobel, 10, 'Sobels Kernels')
    presentFilter(image, Laplacian, 10, 'Laplacian Kernel')

def main(image):
    image = cv.imread(image, cv.IMREAD_GRAYSCALE)
    Prewit = Kernel([[-1 , 0, 1],[-1, 0, 1],[-1, 0, 1]],[[-1, -1, -1],[0, 0, 0],[1, 1, 1]])
    Sobel = Kernel([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]],[[-1, -2, -1],[0,0,0],[1,2,1]])
    Laplacian = Kernel([[0, 1, 0],[1, -4, 1],[0, 1, 0]])
    print("Prewit's Kernels")
    PrewitX = cv.filter2D(image, -1, Prewit.kernels[0])
    PrewitY = cv.filter2D(image, -1, Prewit.kernels[1])
    cv.imshow('Original', image)
    cv.imshow('Prewit X', PrewitX)
    cv.imshow('Prewit Y', PrewitY)
    cv.waitKey()
    cv.destroyAllWindows()
    print("Sobel's Kernels")
    SobelX = cv.filter2D(image, -1, Sobel.kernels[0])
    SobelY = cv.filter2D(image, -1, Sobel.kernels[1])
    cv.imshow('Original', image)
    cv.imshow('Sobel x', SobelX)
    cv.imshow('Sobel y', SobelY)
    cv.waitKey()
    cv.destroyAllWindows()
    print("Laplacian Kernel")
    LaplacianImage = cv.filter2D(image, -1, Laplacian.kernels[0])
    cv.imshow('Original', image)
    cv.imshow('Laplacian', LaplacianImage)
    cv.waitKey()
    cv.destroyAllWindows()
    Gaussian = Kernel([[1, 4, 7, 4, 1],[4, 16, 26, 16, 4],[7, 26, 41, 26, 7],[4, 16, 26, 16, 4],[1, 4, 7, 4, 1]])
    GaussianImage = cv.filter2D(image, -1, (Gaussian.kernels[0] / 273))
    cv.imshow('Original', image)
    cv.imshow('Gaussian', GaussianImage)
    cv.waitKey()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please supply a photo absolute")
    else: 
        if len(sys.argv) < 3:
            main(sys.argv[1])
        elif sys.argv[2] == '-visualisebb':
            visualise(sys.argv[1])
        else:
            print('nah thats wrong usage chef i aint doing that.')