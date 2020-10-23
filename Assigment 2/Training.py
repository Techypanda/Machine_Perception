import os
import time
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
    start = time.time()
    digits = {}
    for dirpath, dirnames, filenames in os.walk("./train/digits"):
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
    end = time.time()
    print("Completed training on SVM model in {} seconds.".format(end - start))

def validation(): # sort out the output name eventually.
    if not os.path.isfile("./digits.dat"):
      raise Exception("SVM Model needs to be trained.")
    svm = cv.ml.SVM_load('./digits.dat')
    for dirpath, dirnames, filenames in os.walk("./val"): break
    valImages = [ dirpath + '/' + filenames[i] for i in range(len(filenames)) ]
    actual = ["48", "35", "94", "302", "71", "26"]
    correct = 0
    start = time.time()
    for i in range(len(valImages)):
        cv.imwrite('./output/validation/DetectedArea{}.jpg'.format(i), digits[1])
        with open("./output/validation/BoundingBox{}.txt".format(i), 'w') as f:
            f.write("{} x, {} y, {} w, {} h".format(str(digits[2][0]), str(digits[2][1]), str(digits[2][2]), str(digits[2][3])))
        reading = ""
        for target in digits[0]:
            gray = cv.resize(cv.cvtColor(target, cv.COLOR_BGR2GRAY), (28,40))
            prediction = int(svm.predict(np.float32(hog(deskew(gray))).reshape(1,-1))[1][0][0])
            reading += str(prediction)
        if reading == actual[i]:
            correct += 1
        with open("./output/validation/House{}.txt".format(i), 'w') as f:
            f.write("Building {}".format(reading))
    end = time.time()
    accuracy = str((float(correct) / float(len(actual))) * 100.0)+"%"
    print("Completed validation in {} seconds.\nAccuracy: {}".format(end - start, accuracy))

def test(): #28 x 40
    extractDigits('./test/tr01.jpg')