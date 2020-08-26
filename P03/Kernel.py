import numpy as np
import cv2 as cv
'''
Kernel Constants
'''
__SOBELS = {
    'x': np.transpose(np.array([
        [-1 , 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ], np.float32)),
    'y': np.transpose(np.array([
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]
    ], np.float32))
}
'''
Application Of Kernels
'''
def applyKernel(image, kernelName):
    if kernelName.upper() == 'SOBEL' or kernelName.upper() == 'SOBELS':
        x = cv.filter2D(image.copy(), -1, __SOBELS['x'])
        y = cv.filter2D(image.copy(), -1, __SOBELS['y'])
        cv.imshow("x", x)
        cv.imshow("y", y)
        return np.sqrt(np.square(x) + np.square(y)).astype(np.float32)