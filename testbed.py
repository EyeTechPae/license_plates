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
mask = np.double(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)) / 255

# frame skipping
skip = 3
frame_count = 0
times = 0

plates = []
width, height = 0, 0

h_x, h_y, h_w, h_h = 0, 0, 0, 0

while True:
    # read and crop image
    _, image = cap.read()

    frame_count += 1
    if frame_count < skip:
        continue
    frame_count = 0

    image = image[100:580, 300:940]

    # mask image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.double(gray)
    gray = cv2.multiply(gray, mask);
    gray = np.uint8(gray)

    # get a list of potential license plates
    segments = localize.get_plate_regions(gray)

    if len(segments) == 1:
        times += 1
    else:
        times = 0

    if times == 8:
        x, y, w, h = segments[0]

        if abs(x-h_x) > 32 or abs(y-h_y) > 32 or abs(w-h_w) > 32 or abs(h-h_h) > 32:
            h_x, x_y, h_w, h_h = x, y, w, h

            plates.append(image[y:y+h, x:x+w])
            cv2.imwrite("plate{}.jpg".format(len(plates)), image[y:y+h, x:x+w])
            #localize.get_plate_regions(gray, debug=image)
            width = max(width, w)
            height += h

    if len(plates) > 0:
        licens = np.zeros((height, width, 3), dtype="uint8")
        acum = 0
        for im in plates:
            shape = im.shape
            licens[acum:acum+shape[0], 0:shape[1]] = im
            acum += shape[0]
        cv2.imshow("licenses", licens)

    # output image
    cv2.imshow("output", image)
    if cv2.waitKey(10 * skip) & 255 == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
