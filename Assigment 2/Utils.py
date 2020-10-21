import cv2 as cv
import numpy as np

def colorReduce(img): # credits to @eliezer-bernart: https://stackoverflow.com/questions/5906693/how-to-reduce-the-number-of-colors-in-an-image-with-opencv/20715062#20715062
    return img // 64 * 64 + 64 // 2

def extractDigits(path):
    img = cv.imread(path, cv.IMREAD_COLOR)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray,(17,15),0)
    edges = cv.Canny(gray, 100, 225)    
    edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, np.ones((4,5), np.uint8))
    edges = cv.dilate(edges, np.ones((4,4), np.uint8))
    image, contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = [ cnt for cnt in contours if cv.contourArea(cnt) > 750 ]
    acceptable = []
    for cnt in contours:
            if len(contours) > 1:
                x,y,w,h = cv.boundingRect(cnt)
                widthHeightRatio = w / h
                heightWidthRatio = h / w
                if w < (img.shape[1] * 0.95):
                    if widthHeightRatio <= 1.5:
                         if heightWidthRatio <= 3.5:
                            acceptable.append(cnt)
            else:
                acceptable.append(cnt)

    if len(acceptable) == 0: # just find a decent plate i guess
        for cnt in contours:
            x,y,w,h = cv.boundingRect(cnt)
            if w < (img.shape[1] * 0.95):
                if h < (img.shape[1] * 0.95):
                    acceptable.append(cnt)
    #cv.drawContours(img, [cnt], 0, (0,255,0), 3)
    
    if len(acceptable) > 1: # find pairs
        pairs = dict()
        for i in range(len(acceptable)):
            for k in range(i+1, len(acceptable)):
                if k < len(acceptable):
                    x,y,w,h = cv.boundingRect(acceptable[i])
                    x2,y2,w2,h2 = cv.boundingRect(acceptable[k])
                    if abs(w - w2) <= 25 and abs(h - h2) <= 20:
                        pairs[i] = acceptable[i]
                        pairs[k] = acceptable[k]
        if len(pairs) > 1:
            print(len(pairs))
            acceptable = pairs.values()
    
    for cnt in contours:
        cv.drawContours(img, [cnt], 0, (0,255,0), 3)
        cv.imshow('contours', img)
        cv.waitKey()

    print(len(acceptable))
    cv.imshow('contours', img)
    cv.imshow('edges', edges)
    cv.waitKey()
    '''
    def detectContours(i, METHOD):
        valid_contours = []
        image, contours, hierarchy = cv.findContours(i, METHOD, cv.CHAIN_APPROX_SIMPLE)
        height_max = img.shape[0] * 0.95
        width_max = img.shape[1] * 0.95
        height_min = img.shape[0] * 0.15
        width_min = img.shape[1] * 0.15
        for cnt in contours:
            x,y,w,h = cv.boundingRect(cnt)
            if w > width_max or w < width_min:
                continue
            if h > height_max or h < height_min:
                continue
            valid_contours.append(cnt)
        
        return valid_contours
    
    img = cv.imread(path)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray,(11,11),0)
    thresh   = cv.threshold(blur, 200, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
    thresh = cv.erode(thresh, np.ones((5,5), np.uint8))
    cv.imshow('thresh', thresh)
    cv.waitKey()

    valid_contours = detectContours(thresh, cv.RETR_EXTERNAL)
    if len(valid_contours) == 1:
        x,y,w,h = cv.boundingRect(valid_contours[0])
        selected = thresh[y:y+h, x:x+w]
        img = img[y:y+h, x:x+w]

        valid_contours_2 = detectContours(selected, cv.RETR_TREE)
        if len(valid_contours_2) != 0:
            valid_contours = valid_contours_2
    if len(valid_contours) == 0:
        valid_contours = detectContours(thresh, cv.RETR_TREE)

    for cnt in valid_contours:
        img = cv.drawContours(img, [cnt], 0, (0,255,0), 3)

    cv.imshow('img', img)
    cv.waitKey()'''
        #if h < image.shape[0] * 0.20 or h > image.shape[0] * 0.80:
        #    break
        #if w < image.shape[1] * 0.15 or w > image.shape[1] * 0.80:
        #    break
        #if cv.contourArea(cnt) > image_area:
        #    img = cv.drawContours(img, [cnt], 0, (0,255,0), 3)
        #    cv.imshow('img', img)
         #   cv.waitKey()
