import cv2
import localize
import sys
import numpy as np

try:
    path = sys.argv[1]
except:
    path = 'test/car.png'

cap = cv2.VideoCapture('entrada2.avi')
mask = cv2.imread('mask.png')
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
mask = np.double(mask)/255;

while True:
    #image = cv2.imread(path)
    _, image = cap.read()
    
    #image = cv2.resize(image, (640, 480))
    x, y = 300, 100
    image = image[y:y+480, x:x+640]

    persp = cv2.getRotationMatrix2D((640/2, 480/2), -15, 1)
    image = cv2.warpAffine(image, persp, (640, 480))

    imagecp = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.double(gray)
    gray = cv2.multiply(gray, mask);
    gray = np.uint8(gray)
    #gray[foreg == 0] = 0

    segments = localize.get_plate_regions(gray)
    plates = []
    width, height = 0, 0
    chars = []
    for i, (x, y, w, h) in enumerate(segments):
        plate = gray[y:y+h, x:x+w]
        digs = localize.get_char_regions(plate)
        cv2.putText(image, 'Hello World!', (x, y-8), cv2.FONT_HERSHEY_DUPLEX, 0.35,  (255,255,0))
        for x1, y1, w1, h1 in digs:
            chars.append(image[y+y1:y+y1+h1, x+x1:x+x1+w1])

        plate = image[y:y+h, x:x+w]
        plates.append(plate)
        width += plate.shape[1]
        height = max(height, plate.shape[0])

    comp = np.zeros((height*2+16, width+8*(len(plates)+1), 3), dtype='uint8')
    off_x = 8
    for plate in plates:
        #plate = cv2.Canny(plate, 100, 120)
        comp[8:8+plate.shape[0], off_x:off_x+plate.shape[1]] = plate
        off_x += plate.shape[1]+8
   
    '''
    off_x = 8
    for char in chars:
        comp[height+8:8+height+char.shape[0], off_x:off_x+char.shape[1]] = char
        off_x += char.shape[1] 
    '''

    composite = np.zeros((480+comp.shape[0], max(640, comp.shape[1]), 3), dtype='uint8')
    composite[0:480, 0:640] = imagecp
    composite[480:480+comp.shape[0], 0:comp.shape[1]] = comp

    cv2.imshow('plates', composite)

    if cv2.waitKey(10) & 255 == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
