import cv2 as cv
import numpy as np

img = cv.imread("./in/digits.png", cv.IMREAD_COLOR)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
digits = np.array([np.hsplit(row, 100) for row in np.vsplit(gray, 50)])
train = digits[:,:50].reshape(-1, 400).astype(np.float32)
test = digits[:,50:100].reshape(-1, 400).astype(np.float32)

k = np.arange(10)
train_labels = np.repeat(k, 250)[:, np.newaxis]
test_labels = train_labels.copy()

knn = cv.ml.KNearest_create()
knn.train(train, cv.ml.ROW_SAMPLE, train_labels)
ret, result, neighbours, dist = knn.findNearest(test, k=1)

matches = result==test_labels
correct = np.count_nonzero(matches)
accuracy = correct*100.0/result.size
print(accuracy)