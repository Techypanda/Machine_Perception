import os
import cv2 as cv
import numpy as np
from Utils import extractDigits

SZ = 40 # height of 40 px for each image.
bin_n = 16
affine_flags = cv.WARP_INVERSE_MAP|cv.INTER_LINEAR
# Chosen learning model: SVM (its fast.) Source: https://github.com/OSSpk/Handwritten-Digits-Classification-Using-KNN-Multiclass_Perceptron-SVM
# https://docs.opencv.org/master/dd/d3b/tutorial_py_svm_opencv.html
# 16 bins
def deskew(img):
    m = cv.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv.warpAffine(img,M,(SZ, SZ),flags=affine_flags)
    return img
def hog(img):
    gx = cv.Sobel(img, cv.CV_32F, 1, 0)
    gy = cv.Sobel(img, cv.CV_32F, 0, 1)
    mag, ang = cv.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
    return hist

def trainDigits():
    digits = {}
    for dirpath, dirnames, filenames in os.walk("./training_data/digits"):
        if len(filenames) > 0:
            digits[dirpath[-1]] = [] # [0 - 9] = ... ROWS
            for image in filenames:
                image = cv.imread(dirpath + "/" + image, 0)
                digits[dirpath[-1]].append(hog(deskew(image)))
    labels = []
    train_data = []
    for key, val in digits.items():
        for x in val:
            labels.append(int(key))
            train_data.append(x)
    labels = np.array(labels).reshape(-1, 1)
    train_data = np.float32(train_data).reshape(-1, 64)
    svm = cv.ml.SVM_create()
    svm.setKernel(cv.ml.SVM_LINEAR)
    svm.setType(cv.ml.SVM_C_SVC)
    svm.setC(2.67)
    svm.setGamma(5.383)
    svm.train(train_data, cv.ml.ROW_SAMPLE, labels)
    svm.save('digits.dat')

def testAccuracy(): #28 x 40
    if not os.path.isfile("./digits.dat"):
      raise Exception("SVM Model needs to be trained.")
    svm = cv.ml.SVM_load('./digits.dat')
    digits = extractDigits("./training_data/number_plates/tr05.jpg")
    #returned = ''
    # print(len(digits))
    '''for digit in digits:
        cv.imshow('t', cv.resize(digit[0], (28, 40)))
        cv.waitKey()
        print('classifying')
        result = np.float32(hog(deskew(cv.resize(digit[0], (28, 40))))).reshape(1, -1)
        test = svm.predict(result)
        print(test)
        cv.destroyAllWindows()'''