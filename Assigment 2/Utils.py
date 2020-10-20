import cv2 as cv
import numpy as np


def extractDigits(path):
    digits = []
    img = cv.imread(path, 0)
    #img = cv.bilateralFilter(img, 15, 200, 200  )
    _, binary = cv.threshold(img, 225, 255, cv.THRESH_OTSU)
    img_output, contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    #contours = []
    # for c in range(len(con)):
    #  if hierarchy[]
    cv.imshow('test', binary)
    cv.waitKey()

    idx = 0
    detected = []
    for cnt in contours:
        # for each contour see if any other imgs are within my size.
        x, y, w, h = cv.boundingRect(cnt)
        detected.append(
            (
                (cnt,
                 (x, y)
                 ),
                (w, h)
            )
        )

    WIDTH_THRESHOLD = 0.95
    HEIGHT_THRESHOLD = 0.95
    ACCEPTABLE_CONSTANT = 0.1
    MIN_VALUE = 25
    print(len(detected))
  
    detected.sort(key=lambda c: c[1][0])
    detected = list(filter(lambda c: not(c[1][0] < MIN_VALUE or c[1][0] > (img.shape[1] * WIDTH_THRESHOLD) or c[1][1] < MIN_VALUE
                                         or c[1][1] > (img.shape[0] * HEIGHT_THRESHOLD)), detected))

    pair = []
    idx = 0
    for d in detected:
        idx = idx + 1
        for d2 in detected[idx:]:
            if abs(d[1][1] - d2[1][1]) <= 15:
                if abs(d[1][0] - d2[1][0]) <= 15:
                    pair.append(d)
                    pair.append(d2)
    print(len(detected))
    if len(pair) < 1:
        detected = detected[-1]
    else:
        detected = pair

    digits = []
    for d in detected:
        digits.append(
            (
                img[
                    d[0][1][1]:d[0][1][1] + d[1][1],
                    d[0][1][0]:d[0][1][0] + d[1][0]
                ],
                (
                    (d[0][1][1], d[0][1][1] + d[1][1]),
                    (d[0][1][0], d[0][1][0] + d[1][0])
                )
            )
        )
        '''cv.imshow('test', img[
      d[0][1][1]:d[0][1][1] + d[1][1],
      d[0][1][0]:d[0][1][0] + d[1][0]    
    ])
    cv.waitKey()
    cv.destroyAllWindows()'''
    return digits

    # return [ (img[cnt[0][1][0] + cnt[1][0], cnt[0][1][1] + cnt[1][1]]) for cnt in detected ]

    # Filter outliers.

    # find the largest pair/s. that dont exceed a certain threshold.

    # if (w >= 30 and w <= 150 and h > 30):
    #    x1 = x+w
    #    y1 = y+h
    #    roi = img[y:y + h, x:x + w]
    #    digits.append(
    #      (roi, ((x, x+w), (y, y+h)))
    #    )
    #digits.sort(key=lambda digit: digit[1][0][0])
    # return digits
