import cv2
import localize
import sys
import numpy as np

try:
    path = sys.argv[1]
except:
    path = 'test/car.png'

cap = cv2.VideoCapture('entrada3.avi')
mask = cv2.imread('mask.png')
mask = np.double(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)) / 255

# frame skipping
skip = 8
frame_count = 0
times = 0

plates = []
width, height = 0, 0

h_x, h_y, h_w, h_h = 0, 0, 0, 0

while True:
    # read and crop image
    try:
        _, image = cap.read()
    except cv2.Error as e:
        break

    frame_count += 1
    if frame_count < skip:
        continue
    frame_count = 0

    image = image[100:580, 300-200:940-200]
    raw = image.copy()

    # mask image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.double(gray)
    gray = cv2.multiply(gray, mask);
    gray = np.uint8(gray)

    # get a list of potential license plates
    segments = localize.get_plate_regions(gray)
    localize.get_plate_regions(gray)

    if len(segments) == 1:
        times += 1
    else:
        times = 0

    if times == 8:
        x, y, w, h = segments[0]

        if abs(x - h_x) > 32 or abs(y - h_y) > 32 or abs(w - h_w) > 32 or abs(h - h_h) > 32:
            h_x, x_y, h_w, h_h = x, y, w, h

            plates.append(image[y:y + h, x:x + w])
            total_im = np.zeros((480, 640, 3), dtype="uint8")
            total_im[y:y + h, x:x + w] = raw[y:y+h, x:x+w]

            #src = np.array([[0, -50], [640, 0+50], [640, 480+50], [0, 480-50]], dtype="float32")
            #dst = np.array([[0, 0], [640, 0], [640, 480], [0, 480]], dtype="float32")
            #M = cv2.getPerspectiveTransform(src, dst)

            #output = np.array((640, 480, 3), dtype="uint8")
            #cv2.perspectiveTransform(total_im, M, output)

            cv2.imwrite("more_plates/plate{}.jpg".format(len(plates)), raw[y:y + h, x:x + w])
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
    if cv2.waitKey(1) & 255 == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
