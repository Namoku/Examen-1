import cv2 as cv
import numpy as np


def cartoonize_image(img, ds_factor=4, sketch_mode=False):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_gray = cv.medianBlur(img_gray, 7)

    edges = cv.Laplacian(img_gray, cv.CV_8U, ksize=5)
    ret, mask = cv.threshold(edges, 100, 255, cv.THRESH_BINARY_INV)

    if sketch_mode:
        return cv.cvtColor(mask, cv.COLOR_GRAY2BGR)

    img_small = cv.resize(img, None, fx=1.0/ds_factor,
                          fy=1.0/ds_factor, interpolation=cv.INTER_AREA)
    num_repetitions = 10
    sigma_color = 5
    sigma_space = 7
    size = 5

    for i in range(num_repetitions):
        img_small = cv.bilateralFilter(
            img_small, size, sigma_color, sigma_space)

    img_output = cv.resize(img_small, None, fx=ds_factor,
                           fy=ds_factor, interpolation=cv.INTER_LINEAR)
    dst = np.zeros(img_gray.shape)

    dst = cv.bitwise_and(img_output, img_output, mask=mask)
    return dst


if __name__ == '__main__':
    capture = cv.VideoCapture(0)

    curChar = -1
    prevChar = -1
    auxKey = ''
    windowName = "Cartoonowo"

    while True:
        ret, frame = capture.read()
        frame = cv.resize(frame, None, fx=0.5, fy=0.5,
                          interpolation=cv.INTER_AREA)
        c = cv.waitKey(1)

        if c == 27:
            break
        if c > -1 and c != prevChar:
            curChar = c
        prevChar = c
        if curChar == ord('s'):
            cv.imshow(windowName, cartoonize_image(frame, sketch_mode=True))
            auxKey = 's'
        elif curChar == ord('c'):
            cv.imshow(windowName, cartoonize_image(frame, sketch_mode=False))
            auxKey = 'c'
        elif curChar == ord('x'):
            if (auxKey == 's'):
                cv.imwrite('cartoon1.png', cartoonize_image(
                    frame, sketch_mode=True))
                auxKey = ''
                curChar = 115
            elif (auxKey == 'c'):
                cv.imwrite('cartoon2.png', cartoonize_image(
                    frame, sketch_mode=False))
                auxKey = ''
                curChar = 99
            else:
                cv.imwrite('photo.png', frame)
                curChar = -1
        else:
            cv.imshow(windowName, frame)
            auxKey = ''

    capture.release()
    cv.destroyAllWindows()
