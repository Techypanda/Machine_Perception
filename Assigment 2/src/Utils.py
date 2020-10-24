import cv2 as cv
import numpy as np
import os

'''
    Purpose: Make a folder, dont care if it fails.
    Date: 24/10/2020
    Author: Jonathan Wright 19779085
'''
def makeFolder(name):
    try:
        os.makedirs(name)
    except Exception as e:
        pass

'''
    Purpose: Performs canny and various blurs + filters to extract and return digits.
    Date: 24/10/2020
    Author: Jonathan Wright 19779085
'''
def extractDigits(path):
    img = cv.imread(path, cv.IMREAD_COLOR)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (15, 17), 0)
    edges = cv.Canny(gray, 115, 160)
    edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    edges = cv.dilate(edges, np.ones((2, 2), np.uint8))
    image, contours, hierarchy = cv.findContours(
        edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv.contourArea(cnt) > 150]
    acceptable = []

    for cnt in contours:
        if len(contours) > 1:
            x, y, w, h = cv.boundingRect(cnt)
            widthHeightRatio = w / h
            heightWidthRatio = h / w
            if w < (img.shape[1] * 0.95):
                if widthHeightRatio <= 1.5:
                    if heightWidthRatio <= 4.8:
                        acceptable.append(cnt)
        else:
            acceptable.append(cnt)
    if len(acceptable) == 1:  # Assume it is a plate and try looking inside
        x, y, w, h = cv.boundingRect(acceptable[0])
        target = img[y:y+h, x:x+w]
        gray = cv.cvtColor(target, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (17, 15), 0)
        edges = cv.Canny(gray, 120, 175)
        edges = cv.morphologyEx(edges, cv.MORPH_CLOSE,
                                np.ones((5, 5), np.uint8))
        edges = cv.dilate(edges, np.ones((2, 2), np.uint8))

        image, contours, hierarchy = cv.findContours(
            edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        contours = [cnt for cnt in contours if cv.contourArea(cnt) > 100]
        acceptable = []
        for cnt in contours:
            if len(contours) > 1:
                x2, y2, w2, h2 = cv.boundingRect(cnt)
                widthHeightRatio = w2 / h2
                heightWidthRatio = h2 / w2
                if w2 < (w * 0.95):
                    if h2 < (h * 0.95):
                        if widthHeightRatio <= 1.5:
                            if heightWidthRatio <= 3.5:
                                acceptable.append(cnt)
            else:
                acceptable.append(cnt)

        detected = []
        for cnt in acceptable:
            x2, y2, w2, h2 = cv.boundingRect(cnt)
            yPos = y+y2
            xPos = x+x2
            detected.append(
                (
                    (xPos, yPos),
                    (w2, h2)
                )
            )
        detected.sort(key=lambda x: x[0][0])
        targets = []
        total_width = 0
        total_height = 0
        if len(detected) == 0:
            return []
        previous_x = detected[0][0][0]
        highest_y = detected[0][0][1]
        for cnt in detected:
            targets.append(
                (img[cnt[0][1]:cnt[0][1]+cnt[1][1], cnt[0]
                     [0]:cnt[0][0]+cnt[1][0]])  # the target
            )
            total_width += cnt[1][0]
            total_height = cnt[1][1] if cnt[1][1] > total_height else total_height
            highest_y = cnt[0][1] if cnt[0][1] < highest_y else highest_y
        plate = img[
            (cnt[0][1]-10):(cnt[0][1]-10)+total_height+10,
            (cnt[0][0]-10):(cnt[0][0]-10)+total_width+10
        ]
        xStart = detected[0][0][0]
        xEnd = detected[-1][0][0]
        widthh = abs(xStart - xEnd) + detected[-1][1][0]
        yPlate = highest_y
        plate = img[yPlate:yPlate+total_height, xStart:xStart + widthh]
        # 0 list of sorted numbers.
        detected = [targets, plate, [xStart, yPlate, widthh, total_height]]
        return detected

    if len(acceptable) == 0:  # just find a decent plate i guess
        for cnt in contours:
            x, y, w, h = cv.boundingRect(cnt)
            if w < (img.shape[1] * 0.95):
                if h < (img.shape[1] * 0.95):
                    acceptable.append(cnt)
    if len(acceptable) > 1:  # find pairs
        pairs = dict()
        for i in range(len(acceptable)):
            for k in range(i+1, len(acceptable)):
                if k < len(acceptable):
                    x, y, w, h = cv.boundingRect(acceptable[i])
                    x2, y2, w2, h2 = cv.boundingRect(acceptable[k])
                    if abs(w - w2) <= 30 and abs(h - h2) <= 15 and abs(y-y2) <= 15 and abs(x - x2) <= (image.shape[1] * 0.30):
                        pairs[i] = acceptable[i]
                        pairs[k] = acceptable[k]
        if len(pairs) > 1:
            acceptable = list(pairs.values())

    not_inside = []  # Check no numbers on same x, as i assume vertical signs will not occur, bad assumption but im doin git
    for i in range(len(acceptable)):
        ni = False
        for k in range(i+1, len(acceptable)):
            if k < len(acceptable):
                x, y, w, h = cv.boundingRect(acceptable[i])
                x2, y2, w2, h2 = cv.boundingRect(acceptable[k])
                if x > x2 and x < x2 + w2:
                    # keep the bigger one.
                    a = w * h
                    a2 = w2 * h2
                    print(a)
                    print(a2)
                    if a < a2:
                        ni = True
        if ni == False:
            not_inside.append(acceptable[i])
    acceptable = not_inside

    contours = []
    for cnt in acceptable:
        x, y, w, h = cv.boundingRect(cnt)
        contours.append(
            (
                (x, y),
                (w, h)
            )
        )

    contours.sort(key=lambda cnt: cnt[0][0])
    targets = []
    total_width = 0
    total_height = 0
    previous_x = contours[0][0][0]
    highest_y = contours[0][0][1]
    for cnt in contours:
        targets.append(
            (img[cnt[0][1]:cnt[0][1]+cnt[1][1], cnt[0]
                 [0]:cnt[0][0]+cnt[1][0]])  # the target
        )
        total_width += cnt[1][0] + abs(previous_x - cnt[0][0])
        total_height = cnt[1][1] if cnt[1][1] > total_height else total_height
        highest_y = cnt[0][1] if cnt[0][1] < highest_y else highest_y
        plate = img[
            (cnt[0][1]-10):(cnt[0][1]-10)+total_height+10,
            (cnt[0][0]-10):(cnt[0][0]-10)+total_width+10
        ]
        xStart = contours[0][0][0]
        xEnd = contours[-1][0][0] + contours[-1][1][0]
        widthh = abs(xEnd - xStart)
        yPlate = highest_y
        plate = img[yPlate:yPlate+total_height, xStart:xStart+widthh]

    # 0 list of sorted numbers.
    detected = [targets, plate, [xStart, yPlate, widthh, total_height]]
    return detected
