import os
import cv2 as cv
import numpy as np

# Chosen learning model: SVM (its fast.) Source: https://github.com/OSSpk/Handwritten-Digits-Classification-Using-KNN-Multiclass_Perceptron-SVM

# https://docs.opencv.org/master/dd/d3b/tutorial_py_svm_opencv.html
# 16 bins


def deskew(img):
    m = cv.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*20*skew], [0, 1, 0]])
    img = cv.warpAffine(
        img, M, (20, 20), flags=cv.WARP_INVERSE_MAP | cv.INTER_LINEAR)
    return img


def hog(img):
    gx = cv.Sobel(img, cv.CV_32F, 1, 0)
    gy = cv.Sobel(img, cv.CV_32F, 0, 1)
    mag, ang = cv.cartToPolar(gx, gy)
    bins = np.int32(16*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    bin_cells = bins[:10, :10], bins[10:, :10], bins[:10, 10:], bins[10:, 10:]
    mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
    hists = [np.bincount(b.ravel(), m.ravel(), 16)
             for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
    return hist
# ^ Opencv docs.


def trainDigits():
    digits = {}
    for dirpath, dirnames, filenames in os.walk("./training_data/digits"):
        if len(filenames) > 0:
            digits[dirpath[-1]] = []
            for file in filenames:
              digits[dirpath[-1]].append(
                cv.imread(dirpath + '/' + file, 0)
              )
    
    npArrayLength = 0
    for label, images in digits.items():
      npArrayLength += len(images)
    
    labels = []
    train_data = []

    for label, images in digits.items():
      deskewed = [list(map(deskew, img)) for img in images]
      h = [list(map(hog, img)) for img in deskewed]
      for i in range(len(h)):
        labels.append(int(label))
        train_data.append(h[i])
    
    train_data = np.float32(train_data).reshape(-1, 132)
    labels = np.array(labels).reshape(132, 1)
    print(train_data)
    print(len(labels))

    svm = cv.ml.SVM_create()
    svm.setKernel(cv.ml.SVM_LINEAR)
    svm.setType(cv.ml.SVM_C_SVC)
    svm.setC(2.67)
    svm.setGamma(5.383)

    svm.train(train_data, cv.ml.ROW_SAMPLE, labels)
    svm.save('test.dat')
