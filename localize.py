import cv2
import numpy as np

# convolution kernel used to compute density
_density_kernel = np.ones((3, 45), dtype='float64')

def get_char_regions (image):
    '''Look for regions that resemble license plate chracters'''
    
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) 
    image = cv2.GaussianBlur(image, (3, 3), 3)
    #image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, ker)

    # threshold image and segment it
    thr = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 12)
    _, conts, hier = cv2.findContours(thr.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    potentials = []
    for i, cont in enumerate(conts):
        x, y, w, h = cv2.boundingRect(cont)
        aspect = w/h
        if aspect > 0.8 or w > h or w > 32 or h > 42 or w < 8 or h < 16:
            continue
        
        potentials.append((x, y, w, h))
    
    #cv2.imshow('chars', thr)
    #cv2.waitKey(0)

    return potentials

def get_plate_regions (image, debug=None):
    regions = get_plate_regions_pre(image)
    plates = []
    for x, y, w, h in regions:
        if debug is not None:
            cv2.rectangle(debug, (x, y), (x+w, y+h), (0, 255, 255), thickness=1)
        plate = image[y:y+h, x:x+w]
        digs = get_char_regions(plate)
        #print(len(digs))
        if debug is not None:
            for x0, y0, w0, h0 in digs:
                cv2.rectangle(debug, (x+x0, y+y0), (x+x0+w0, y+y0+h0), (255, 0, 255), thickness=1)            

        if len(digs) <= 4:
            continue
        min_y, max_y = 9999, -9999
        min_x, max_x = 9999, -9999
        for x0, y0, w0, h0 in digs:
            #print(x0, y0, w0, h0)
            cx, cy = x0+w0/2, y0+h0/2
            min_y, max_y = min(min_y, cy), max(max_y, cy)
            min_x, max_x = min(min_x, cx), max(max_x, cx)

       
        dif_y = max(0, max_y-min_y)
        dif_x = max(0, max_x-min_x + 32)
        
        if dif_x < 64:
            continue
        
        if (dif_x > 64 or dif_x > w*0.5) and dif_y < 64:
            plates.append((x, y, w, h))
            
            if debug is not None:
                cv2.rectangle(debug, (x, y), (x+w, y+h), (255, 0, 255), thickness=2)
    
    
    return plates 

def get_plate_regions_pre (image):
    '''Compute a list of potential license plates'''

    # top hat
    #image = cv2.GaussianBlur(image, (5, 5), 4)
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, ker)
    top = cv2.subtract(closed, image)
    tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, ker)
    gray = tophat

    # sobel operator over preprocessed image
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0)
    #sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1)
    sobel_x[sobel_x < 0] = -sobel_x[sobel_x < 0]
    #sobel_y[sobel_y < 0] = -sobel_y[sobel_y < 0]
    sobel_add = sobel_x
    sobel_add /= sobel_add.max() 
    sobel = sobel_add.copy()
    sobel_add[sobel_add <= 0.25] = 0
    sobel_add[sobel_add > 0.25] = 1
    

    # compute density
    dens = cv2.filter2D(sobel_add, cv2.CV_64F, _density_kernel)
    #ker = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9))
    #dens = cv2.morphologyEx(dens, cv2.MORPH_DILATE, ker)
    #dens = cv2.GaussianBlur(dens, (5, 9), 7)
    
    # threshold density
    dens = dens / dens.max()
    dens[dens <= 0.3] = 0
    dens[dens > 0.3] = 255
    dens = np.uint8(dens)

    # find contours in image
    _, conts, _ = cv2.findContours(dens.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    potentials = []
    for cont in conts:
        x, y, w, h = cv2.boundingRect(cont)
        if w > 300 or w < 64 or h > 200 or h < 16:
            continue
        if w/h > 1 and w/h < 8:
            e = 8 
            potentials.append((max(x-e, 0), max(y-e, 0), min(w+e*2, 640), min(h+e*2, 480))) 

    #cv2.imshow("step", sobel)
    #cv2.waitKey(0)
    
    return potentials
